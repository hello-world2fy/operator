"""Blind objective code. NO oracle or post-hoc diagnostics imports.

Inputs: the operator/state, local Hilbert representation, geometric gate
supports, generic initialization and a fixed evaluation budget. No known
Marshall, loop, domain-wall, Huse-Elser or path labels are accepted.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from ..optimize import batched_moment, _adam_batch, _lbfgs_one, riemannian_gradient
from ..walsh import intrinsic_alpha0,walshness_from_directions


def blind_starts(n,starts,rng,alpha0=None):
    axes=rng.normal(size=(starts,n,3));axes/=np.linalg.norm(axes,axis=-1,keepdims=True)
    axes[0]=[1,0,0]  # uniform, no staggered or 120-degree template
    if alpha0 is not None and starts>1:axes[1]=alpha0
    return axes


def alpha0_from_hs_rows(operator):
    """Hermitian alpha0 from HS row marginals in O(n d^2), avoiding dense O^2."""
    operator=np.asarray(operator)
    if not np.allclose(operator,operator.conj().T,atol=1e-12):raise ValueError('Hermitian operator required')
    d=len(operator);n=int(np.log2(d));states=np.arange(d)
    norm=float(np.vdot(operator,operator).real);weights=np.sum(abs(operator)**2,axis=1)/norm
    bloch=np.zeros((n,3))
    for i in range(n):
        z=1.-2*((states>>i)&1)
        cross=np.sum(operator.conj()*operator[states^(1<<i)],axis=1)/norm
        bloch[i]=[cross.sum().real,np.sum(-1j*z*cross).real,weights@z]
    lengths=np.linalg.norm(bloch,axis=1)
    axes=np.tile([1.,0.,0.],(n,1));nonzero=lengths>1e-12
    axes[nonzero]=bloch[nonzero]/lengths[nonzero,None]
    return float(n-lengths.sum()),axes


def optimize_local(operator,alpha,*,starts=24,steps=150,polish=30,seed=0,device='cpu'):
    begin=time.monotonic();n=int(np.log2(len(operator)))
    w0,axes0=alpha0_from_hs_rows(operator) if n>=12 else intrinsic_alpha0(operator)
    initial=blind_starts(n,starts,np.random.default_rng(seed),axes0)
    op=torch.tensor(operator,dtype=torch.complex128,device=device)
    # Explicit n>=12 chunk guard: the full autograd tape is much larger than
    # the input matrix. Two restarts fit comfortably on an 80 GiB A100.
    chunk=2 if n>=12 else starts
    final=np.concatenate([_adam_batch(op,initial[i:i+chunk],alpha,steps,.08,torch.device(device))
                          for i in range(0,starts,chunk)])
    with torch.no_grad():
        values=torch.cat([torch.log(batched_moment(op,torch.tensor(final[i:i+chunk],device=device),alpha))/alpha
                          for i in range(0,starts,chunk)])
    for index in torch.argsort(values)[:min(4,starts)].cpu().numpy():
        final[index]=_lbfgs_one(op,final[index],alpha,torch.device(device),polish)
    with torch.no_grad():
        values=torch.cat([torch.log(batched_moment(op,torch.tensor(final[i:i+chunk],device=device),alpha))/alpha
                          for i in range(0,starts,chunk)]).cpu().numpy()
    best=int(np.argmin(values));gradient=riemannian_gradient(op,final[best],alpha,torch.device(device))
    checked=walshness_from_directions(operator,final[best],alpha)
    if abs(checked-values[best])>1e-8:raise ArithmeticError('moment convention cross-check failed')
    return {'kind':'local_SU2','alpha':alpha,'seed':seed,'initial_axes':initial,'axes_restarts':final,
            'W_restarts':values,'best_index':best,'axes_best':final[best],'W_best':checked,
            'gradient_norm':np.linalg.norm(gradient,axis=-1),'alpha0_W':w0,'alpha0_axes':axes0,
            'steps':steps,'polish_steps':polish,'seconds':time.monotonic()-begin,
            'certified_global':False,'objective':'Operator_Walshness_only'}


def fwht_torch_flat(x):
    """Normalized FWHT of last dimension, for non-autograd objective evaluations."""
    x=x.clone();d=x.shape[-1];stride=1
    if d&(d-1):raise ValueError('power-of-two transform required')
    while stride<d:
        a=x.reshape(*x.shape[:-1],d//(2*stride),2,stride)
        left=a[...,0,:].clone();right=a[...,1,:]
        a[...,0,:]=left+right;a[...,1,:]=left-right
        stride*=2
    return x/np.sqrt(d)


def gate_features(n,supports,symmetric=False):
    """Truth tables of local gates. Symmetric gates commute with product X on
    their OWN support. For edges/triangles, use complement-paired projectors
    (-1)^[bits all zero OR all one]; single-site symmetric diagonal gates are
    identity and excluded. This is a stated restricted family, not all symmetric
    circuits. Gatewise symmetry is checked rather than inferred globally.
    """
    states=np.arange(1<<n,dtype=np.int64);features=[];used=[]
    for support in supports:
        if symmetric and len(support)<2:continue
        mask=sum(1<<int(i) for i in support)
        feature=(states&mask)==mask
        if symmetric:feature|=(states&mask)==0
        features.append(feature);used.append(tuple(support))
    return np.array(features,dtype=np.int8),used


def circuit_depth(supports,coefficients):
    # Greedy parallel scheduling for commuting diagonal gates.
    layers=[]
    for support,on in zip(supports,coefficients):
        if not on:continue
        for layer in layers:
            if not layer.intersection(support):layer.update(support);break
        else:layers.append(set(support))
    return len(layers)


def optimize_circuit(operator,alpha,supports,*,starts=16,flips=2000,seed=0,
                     max_gates=None,max_depth=None,symmetric=False,device='cpu',pure=False):
    begin=time.monotonic();operator=np.asarray(operator);n=int(np.log2(len(operator)))
    features,supports=gate_features(n,supports,symmetric)
    occupied=np.flatnonzero(operator) if pure else None
    restricted=bool(pure and len(occupied)<len(operator)//4)
    if restricted:features=features[:,occupied]
    if len(supports)==0:raise ValueError('empty nontrivial gate family')
    rng=np.random.default_rng(seed);coeff=np.zeros((starts,len(supports)),dtype=np.int8)
    max_gates=min(len(supports),n if max_gates is None else max_gates)
    max_depth=n if max_depth is None else max_depth
    for k in range(1,starts):
        for index in rng.permutation(len(supports))[:rng.integers(0,max_gates+1)]:
            coeff[k,index]=1
            if circuit_depth(supports,coeff[k])>max_depth:coeff[k,index]=0
    initial=coeff.copy();ft=torch.tensor(1.-2*features,dtype=torch.float64,device=device)
    op=torch.tensor(operator[occupied] if restricted else operator,
                    dtype=torch.complex128 if np.iscomplexobj(operator) else torch.float64,device=device)
    parity=(coeff.astype(np.int64)@features)%2
    phase=torch.tensor(1.-2*parity,dtype=torch.float64,device=device)
    degree=np.bitwise_count(np.arange(1<<n,dtype=np.uint64)).astype(float)
    if not pure:degree=(degree[:,None]+degree[None,:]).ravel()
    shift=n if pure else 2*n
    weight=torch.tensor(np.exp(alpha*(degree-shift)),dtype=torch.float64,device=device)
    norm=float(np.vdot(operator,operator).real)
    if restricted:
        distance=np.bitwise_count((occupied[:,None]^occupied[None,:]).astype(np.uint64))
        a=(1+np.exp(alpha))/2;b=(1-np.exp(alpha))/2
        kernel=torch.tensor(a**(n-distance)*b**distance,dtype=op.dtype,device=device)
    def objective(phases):
        if restricted:
            values=op*phases
            moment=(values.conj()*(values@kernel.T)).sum(-1).real/norm
            return 2*torch.log(moment)/alpha
        values=op*phases if pure else op[None,:,:]*phases[:,:,None]*phases[:,None,:]
        transformed=fwht_torch_flat(values.reshape(starts,-1))
        moment=(abs(transformed)**2*weight).sum(-1)/norm
        w=torch.log(moment)/alpha+shift
        return 2*w if pure else w
    with torch.no_grad():
        current=objective(phase);best_values=current.clone();best=coeff.copy()
        # Candidate proposals are independent of all labels. Simulated annealing
        # in the first half, zero-temperature coordinate search in the second.
        for step in range(flips):
            proposal=rng.integers(len(supports),size=starts)
            allowed=np.ones(starts,bool)
            candidate_coeff=coeff.copy();candidate_coeff[np.arange(starts),proposal]^=1
            for k in range(starts):
                allowed[k]=(candidate_coeff[k].sum()<=max_gates and circuit_depth(supports,candidate_coeff[k])<=max_depth)
            proposed_phase=phase*ft[torch.tensor(proposal,device=device)]
            new=objective(proposed_phase)
            temperature=.04*max(0,1-2*step/max(flips,1))
            diff=(new-current).cpu().numpy()
            accept=allowed&((diff<0)|(rng.random(starts)<np.exp(np.minimum(0,-diff/max(temperature,1e-15)))))
            accept_t=torch.tensor(accept,device=device)
            current=torch.where(accept_t,new,current);phase=torch.where(accept_t[:,None],proposed_phase,phase)
            coeff[accept]=candidate_coeff[accept]
            improve=(current<best_values-1e-13).cpu().numpy()
            best_values=torch.minimum(current,best_values);best[improve]=coeff[improve]
    best_values=best_values.cpu().numpy();index=int(np.argmin(best_values))
    return {'kind':'diagonal_circuit','alpha':alpha,'seed':seed,'gate_supports':supports,
            'gate_family':'complement_paired_diagonal' if symmetric else 'Z_CZ_CCZ',
            'gatewise_Z2_symmetric':symmetric,'initial_coefficients':initial,'coefficients_restarts':best,
            'coefficients_best':best[index],'W_restarts':best_values,'W_best':float(best_values[index]),
            'gate_count':int(best[index].sum()),'circuit_depth':circuit_depth(supports,best[index]),
            'max_gates':max_gates,'max_depth':max_depth,'flips_per_start':flips,
            'evaluations':starts*(flips+1),'seconds':time.monotonic()-begin,
            'pure_projector_objective':pure,'objective':'Operator_Walshness_only',
            'certified_global':False,'backend':'exact_supported_pure_kernel' if restricted else 'full_FWHT'}


def circuit_phase(n,result):
    features,_=gate_features(n,result['gate_supports'],result['gatewise_Z2_symmetric'])
    return 1.-2*((np.asarray(result['coefficients_best'],dtype=np.int64)@features)%2)


def qutrit_moment(operator,digits,alpha,unitaries=None):
    """Restricted-kernel exact Z3 support-degree moment without a 3^n x 3^n rho.

    For embedded O=P O P, Tr(O† A O A) uses only P A P. F† diag(1,e^a,e^a) F
    is the local factor; basis rotations are U1_hole direct-sum SU2_spin.
    """
    n=digits.shape[1];local=np.exp(alpha)*np.eye(3)+(1-np.exp(alpha))*np.ones((3,3))/3
    kernel=np.ones((len(digits),len(digits)),dtype=complex)
    for i in range(n):
        a=local if unitaries is None else unitaries[i]@local@unitaries[i].conj().T
        kernel*=a[digits[:,i,None],digits[None,:,i]]
    moment=np.vdot(operator,kernel@operator@kernel).real/np.vdot(operator,operator).real
    return float(moment)


def pure_moment_torch(psi,axes,alpha):
    from ..optimize import _local_factors
    n=axes.shape[1];batch=len(axes);d=psi.numel()
    local=_local_factors(axes,alpha)
    x=psi[None,:].expand(batch,-1)
    for site in range(n):
        stride=1<<site;x=x.reshape(batch,d//(2*stride),2,stride)
        x=torch.einsum('bno,bhol->bhnl',local[:,site],x).reshape(batch,d)
    return torch.einsum('j,bj->b',psi.conj(),x).real/torch.vdot(psi,psi).real


def supported_pure_moment(psi,configurations,axes,alpha):
    from ..optimize import _local_factors
    factors=_local_factors(axes,alpha);bits=(configurations[:,None]>>torch.arange(axes.shape[1],device=axes.device))&1
    kernel=torch.ones((len(axes),len(psi),len(psi)),dtype=torch.complex128,device=axes.device)
    for i in range(axes.shape[1]):
        kernel=kernel*factors[:,i,bits[:,i,None],bits[None,:,i]]
    return torch.einsum('i,bij,j->b',psi.conj(),kernel,psi).real/torch.vdot(psi,psi).real


def optimize_pure_local(psi,alpha,*,starts=24,steps=150,seed=0,device='cpu'):
    """Pure-state carryover, using O(n 2^n) contractions; never form |psi><psi|."""
    begin=time.monotonic();psi=np.asarray(psi,dtype=complex);n=int(np.log2(len(psi)))
    rng=np.random.default_rng(seed);initial=blind_starts(n,starts,rng)
    occupied=np.flatnonzero(psi);restricted=len(occupied)<len(psi)//4
    vector=torch.tensor(psi[occupied] if restricted else psi,dtype=torch.complex128,device=device)
    states=torch.tensor(occupied,dtype=torch.int64,device=device)
    moment=(lambda axes:supported_pure_moment(vector,states,axes,alpha)) if restricted else (lambda axes:pure_moment_torch(vector,axes,alpha))
    raw=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float64,device=device))
    optimizer=torch.optim.Adam([raw],lr=.06)
    best_values=np.full(starts,np.inf);best=initial.copy()
    for _ in range(steps):
        optimizer.zero_grad();axes=raw/raw.norm(dim=-1,keepdim=True)
        w=2*torch.log(moment(axes))/alpha
        w.sum().backward();optimizer.step()
        values=w.detach().cpu().numpy();improved=values<best_values
        best_values[improved]=values[improved];best[improved]=axes.detach().cpu().numpy()[improved]
    index=int(np.argmin(best_values))
    raw=torch.tensor(best[index],dtype=torch.float64,device=device,requires_grad=True)
    ival=2*torch.log(moment((raw/raw.norm(dim=-1,keepdim=True))[None,:,:]))/alpha
    ival.sum().backward()
    return {'kind':'pure_local_SU2','alpha':alpha,'seed':seed,'initial_axes':initial,
            'axes_restarts':best,'axes_best':best[index],'W_restarts':best_values,
            'W_best':float(best_values[index]),'gradient_norm':raw.grad.norm(dim=-1).detach().cpu().numpy(),
            'pure_projector_objective':True,'objective':'Operator_Walshness_only',
            'steps':steps,'seconds':time.monotonic()-begin,'certified_global':False,
            'backend':'exact_supported_pure_kernel' if restricted else 'full_wavefunction_contractions'}


def rotate_pure(psi,unitaries):
    out=np.asarray(psi,dtype=complex).copy();d=len(out)
    for site,u in enumerate(unitaries):
        stride=1<<site
        out=np.einsum('ab,hbl->hal',u.conj().T,out.reshape(d//(2*stride),2,stride)).reshape(d)
    return out


def qutrit_local_vectors(raw):
    """SU2 spin rotation of |up>+|down>, hole amplitude fixed; all allowed
    rank-one Fourier directions are covered including the relative hole phase.
    """
    q=raw/raw.norm(dim=-1,keepdim=True).clamp_min(1e-15)
    a=q[...,0]+1j*q[...,1];b=q[...,2]+1j*q[...,3]
    return torch.stack([torch.ones_like(a),(a+b),(-b.conj()+a.conj())],dim=-1)/np.sqrt(3)


def qutrit_batched_moment(op,digits,alpha,vectors):
    batch,n,_=vectors.shape;d=len(op)
    kernel=torch.ones((batch,d,d),dtype=torch.complex128,device=op.device)
    for i in range(n):
        v=vectors[:,i];local=(1-np.exp(alpha))*v[:,:,None]*v[:,None,:].conj()
        local=local+np.exp(alpha)*torch.eye(3,device=op.device)
        kernel=kernel*local[:,digits[:,i,None],digits[None,:,i]]
    transformed=kernel@op@kernel
    return torch.einsum('ij,bij->b',op.conj(),transformed).real/torch.vdot(op.ravel(),op.ravel()).real


def optimize_qutrit_local(operator,digits,alpha,*,starts=24,steps=150,seed=0,device='cpu'):
    begin=time.monotonic();rng=np.random.default_rng(seed);n=digits.shape[1]
    initial=rng.normal(size=(starts,n,4));initial/=np.linalg.norm(initial,axis=-1,keepdims=True)
    initial[0]=[1,0,0,0]
    raw=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float64,device=device))
    op=torch.tensor(operator,dtype=torch.complex128,device=device)
    configs=torch.tensor(digits.astype(np.int64),device=device)
    optimizer=torch.optim.Adam([raw],lr=.05);best=initial.copy();best_values=np.full(starts,np.inf)
    for _ in range(steps):
        optimizer.zero_grad();vectors=qutrit_local_vectors(raw)
        w=torch.log(qutrit_batched_moment(op,configs,alpha,vectors))/alpha
        values=w.detach().cpu().numpy();improved=values<best_values
        snapshot=(raw/raw.norm(dim=-1,keepdim=True)).detach().cpu().numpy().copy()
        best_values[improved]=values[improved];best[improved]=snapshot[improved]
        w.sum().backward();optimizer.step()
    # Recompute every stored quaternion; do not return a score without its state.
    with torch.no_grad():
        best_values=(torch.log(qutrit_batched_moment(op,configs,alpha,qutrit_local_vectors(torch.tensor(best,device=device))))/alpha).cpu().numpy()
    index=int(np.argmin(best_values))
    return {'kind':'qutrit_hole_U1_spin_SU2','initial_quaternions':initial,'quaternions_restarts':best,
            'quaternions_best':best[index],'W_restarts':best_values,'W_best':float(best_values[index]),
            'alpha':alpha,'seed':seed,'steps':steps,'seconds':time.monotonic()-begin,
            'metric':'Z3_frequency_support_degree','objective':'Operator_Walshness_only',
            'certified_global':False}
