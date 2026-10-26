"""Exact sparse-Hermitian moment backend. No physical sign references.

For sparse Hermitian O, X=O A and Tr(O A O A)=sum_ij X_ij X_ji.
Explicit product A is cheaper than applying 2n strided contractions to O.
This changes the contraction order, not the objective or density matrix.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from scipy import sparse
from ..optimize import _local_factors
from .blind import blind_starts,gate_features,circuit_depth


def sparse_operator(operator,device):
    coo=sparse.coo_matrix(operator)
    if abs((coo-coo.conj().T).data).max(initial=0.)>1e-11:raise ValueError('Hermitian input required')
    ids=torch.tensor(np.stack([coo.row,coo.col]),dtype=torch.int64,device=device)
    values=torch.tensor(coo.data,dtype=torch.complex128,device=device)
    return torch.sparse_coo_tensor(ids,values,coo.shape,device=device).coalesce()


def explicit_product_factors(axes,alpha):
    factors=_local_factors(axes,alpha);b,n=axes.shape[:2]
    result=torch.ones((b,1,1),dtype=torch.complex128,device=axes.device)
    for i in range(n-1,-1,-1):
        d=result.shape[-1]
        result=torch.einsum('bij,bkl->bikjl',result,factors[:,i]).reshape(b,2*d,2*d)
    return result


def sparse_hermitian_moment(operator,axes,alpha):
    b=axes.shape[0];d=operator.shape[0];a=explicit_product_factors(axes,alpha)
    flattened=a.permute(1,0,2).reshape(d,b*d)
    product=torch.sparse.mm(operator,flattened).reshape(d,b,d).permute(1,0,2)
    numerator=(product*product.transpose(1,2)).sum((1,2)).real
    norm=(abs(operator.values())**2).sum()
    return numerator/norm


class SmallOffdiagonalMoment:
    """Exact O=D+S contraction when off-diagonal entries touch few states.

    Tr(O A O A)=Tr(D A D A)+2 Tr(S A D A)+Tr(S A S A).
    D is kept on the full Hilbert space; only S has restricted support. No
    state, diagonal weight, thermal sector or matrix entry is discarded.
    """
    def __init__(self,operator,device):
        op=sparse.csr_matrix(operator);off=op.copy();off.setdiag(0);off.eliminate_zeros()
        self.active=np.unique(off.nonzero()[0]);self.n=int(np.log2(op.shape[0]));self.device=device
        self.diagonal=torch.tensor(op.diagonal().real,dtype=torch.float64,device=device)
        self.s=torch.tensor(off[self.active][:,self.active].toarray(),dtype=torch.complex128,device=device)
        self.norm=float(np.vdot(op.data,op.data).real)
        self.all_bits=torch.tensor((np.arange(op.shape[0])[:,None]>>np.arange(self.n))&1,device=device)
        self.active_ids=torch.tensor(self.active,device=device)
        self.active_bits=self.all_bits[self.active_ids]

    def product_rows(self,factors,columns):
        # Kronecker rows use reduction gradients, avoiding millions of atomic
        # scatter-adds into the same four local factor entries.
        result=torch.ones((len(factors),len(self.active),1),dtype=torch.complex128,device=self.device)
        for i in range(self.n-1,-1,-1):
            local=factors[:,i].index_select(1,self.active_bits[:,i])
            result=torch.einsum('bri,brj->brij',result,local).reshape(len(factors),len(self.active),-1)
        if len(columns)==len(self.all_bits):return result
        ids=(columns*(2**torch.arange(self.n,device=self.device))).sum(1)
        return result.index_select(2,ids)

    def diagonal_part(self,factors):
        x=self.diagonal[None,:].expand(len(factors),-1);d=len(self.diagonal)
        for i in range(self.n):
            stride=1<<i;x=x.reshape(len(factors),d//(2*stride),2,stride)
            x=torch.einsum('bao,bhol->bhal',abs(factors[:,i])**2,x).reshape(len(factors),d)
        return (x*self.diagonal).sum(-1)

    def __call__(self,axes,alpha):
        factors=_local_factors(axes,alpha);constant=self.diagonal_part(factors)
        if not len(self.active):return constant/self.norm
        rows=self.product_rows(factors,self.all_bits);a=rows.index_select(2,self.active_ids)
        product=self.s@a
        quadratic=(product*product.transpose(1,2)).sum((1,2)).real
        cross=((self.s@rows)*rows.conj()*self.diagonal).sum((1,2)).real
        return (constant+2*cross+quadratic)/self.norm


class SmallOffdiagonalPhaseObjective:
    def __init__(self,operator,alpha,batch,device):
        structure=SmallOffdiagonalMoment(operator,device);self.alpha=alpha;self.norm=structure.norm
        self.active=torch.tensor(structure.active,device=device);self.s=structure.s
        axes=torch.zeros((1,structure.n,3),dtype=torch.float64,device=device);axes[:,:,0]=1
        with torch.no_grad():
            factors=_local_factors(axes,alpha)
            self.a=structure.product_rows(factors,structure.active_bits)[0]
            rows=structure.product_rows(factors,structure.all_bits)[0]
            self.cross=(rows*structure.diagonal)@rows.conj().T
            self.constant=structure.diagonal_part(factors)[0]

    def __call__(self,phases):
        phi=phases[:,self.active]
        s=self.s*phi[:,:,None]*phi[:,None,:]
        product=s@self.a
        numerator=self.constant+2*(s*self.cross.T).sum((1,2)).real+(product*product.transpose(1,2)).sum((1,2)).real
        return torch.log(numerator/self.norm)/self.alpha


def optimize_sparse_local(operator,alpha,*,starts=24,steps=150,polish=20,seed=0,device='cpu',chunk=4):
    began=time.monotonic();coo=sparse.csr_matrix(operator);n=int(np.log2(coo.shape[0]))
    # Alpha0 one-site HS marginals can be computed from sparse rho^2 exactly.
    squared=(coo@coo).tocsr();norm=float(np.vdot(coo.data,coo.data).real)
    states=np.arange(coo.shape[0]);bloch=[]
    for i in range(n):
        z=1.-2*((states>>i)&1);flipped=states^(1<<i)
        off=np.asarray(squared[states,flipped]).ravel()
        bloch.append([off.sum().real/norm,np.sum(1j*z*off).real/norm,float(squared.diagonal().real@z/norm)])
    bloch=np.asarray(bloch);norms=np.linalg.norm(bloch,axis=1)
    axes0=np.tile([1.,0.,0.],(n,1));good=norms>1e-12;axes0[good]=bloch[good]/norms[good,None]
    initial=blind_starts(n,starts,np.random.default_rng(seed),axes0)
    structure=SmallOffdiagonalMoment(coo,device)
    if len(structure.active)<coo.shape[0]//2:
        moment=lambda axes:structure(axes,alpha)
        backend='exact_full_diagonal_plus_small_offdiagonal_support'
    else:
        op=sparse_operator(coo,device);moment=lambda axes:sparse_hermitian_moment(op,axes,alpha)
        backend='explicit_product_A_sparse_OA_trace_square'
    endpoints=[];values=[]
    for start in range(0,starts,chunk):
        raw=torch.nn.Parameter(torch.tensor(initial[start:start+chunk],dtype=torch.float64,device=device))
        optim=torch.optim.Adam([raw],lr=.08);best=np.full(len(raw),np.inf);best_axes=initial[start:start+chunk].copy()
        for _ in range(steps):
            optim.zero_grad();axes=raw/raw.norm(dim=-1,keepdim=True)
            w=torch.log(moment(axes))/alpha
            current=w.detach().cpu().numpy();improved=current<best
            best[improved]=current[improved];best_axes[improved]=axes.detach().cpu().numpy()[improved]
            w.sum().backward();optim.step()
        endpoints.extend(best_axes);values.extend(best)
    endpoints=np.array(endpoints);values=np.array(values)
    for index in np.argsort(values)[:min(4,starts)]:
        raw=torch.nn.Parameter(torch.tensor(endpoints[index],dtype=torch.float64,device=device))
        optimizer=torch.optim.LBFGS([raw],max_iter=polish,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
        def closure():
            optimizer.zero_grad();axes=raw/raw.norm(dim=-1,keepdim=True)
            w=torch.log(moment(axes[None,:,:]))[0]/alpha;w.backward();return w
        optimizer.step(closure)
        with torch.no_grad():
            axes=raw/raw.norm(dim=-1,keepdim=True);candidate=float(torch.log(moment(axes[None,:,:]))[0]/alpha)
        if candidate<values[index]:endpoints[index]=axes.cpu().numpy();values[index]=candidate
    index=int(np.argmin(values));raw=torch.tensor(endpoints[index],dtype=torch.float64,device=device,requires_grad=True)
    w=torch.log(moment((raw/raw.norm(dim=-1,keepdim=True))[None,:,:]))[0]/alpha;w.backward()
    return {'kind':'local_SU2','objective':'Operator_Walshness_only','alpha':alpha,'seed':seed,
        'initial_axes':initial,'axes_restarts':endpoints,'axes_best':endpoints[index],
        'W_restarts':values,'W_best':float(values[index]),'gradient_norm':raw.grad.norm(dim=-1).detach().cpu().numpy(),
        'alpha0_axes':axes0,'alpha0_W':float(n-norms.sum()),'steps':steps,'polish_steps':polish,
        'seconds':time.monotonic()-began,'exact_sparse_input':True,'nonzero_entries':coo.nnz,
        'certified_global':False,'backend':backend}


class SparsePhaseObjective:
    """Block-diagonal batching of sparse D O D times a fixed product A_X."""
    def __init__(self,operator,alpha,batch,device):
        coo=sparse.csr_matrix(operator).tocoo();self.batch=batch;self.d=coo.shape[0];self.alpha=alpha
        self.n=int(np.log2(self.d));self.device=device
        real=abs(coo.data.imag).max(initial=0.)<1e-14
        dtype=torch.float64 if real else torch.complex128
        values=coo.data.real if real else coo.data
        self.values=torch.tensor(values,dtype=dtype,device=device)
        self.row=torch.tensor(coo.row,dtype=torch.int64,device=device)
        self.col=torch.tensor(coo.col,dtype=torch.int64,device=device)
        offset=(np.arange(batch)*self.d)[:,None]
        ids=np.stack([(coo.row[None,:]+offset).ravel(),(coo.col[None,:]+offset).ravel()])
        self.ids=torch.tensor(ids,dtype=torch.int64,device=device)
        axes=torch.zeros((1,self.n,3),dtype=torch.float64,device=device);axes[:,:,0]=1
        a=explicit_product_factors(axes,alpha)[0]
        self.a=(a.real if real else a).repeat(batch,1)
        self.norm=float(np.vdot(coo.data,coo.data).real)

    def __call__(self,phases):
        values=(phases[:,self.row]*phases[:,self.col]*self.values).ravel()
        operator=torch.sparse_coo_tensor(self.ids,values,(self.batch*self.d,)*2,
                                         is_coalesced=True,device=self.device)
        product=torch.sparse.mm(operator,self.a).reshape(self.batch,self.d,self.d)
        moment=(product*product.transpose(1,2)).sum((1,2)).real/self.norm
        return torch.log(moment)/self.alpha


def optimize_sparse_circuit(operator,alpha,supports,*,starts=16,flips=2000,seed=0,max_gates=None,max_depth=None,
                            symmetric=False,device='cpu'):
    begin=time.monotonic();n=int(np.log2(operator.shape[0]));features,supports=gate_features(n,supports,symmetric)
    rng=np.random.default_rng(seed);coeff=np.zeros((starts,len(supports)),dtype=np.int8)
    max_gates=min(len(supports),n if max_gates is None else max_gates);max_depth=n if max_depth is None else max_depth
    for k in range(1,starts):
        for index in rng.permutation(len(supports))[:rng.integers(0,max_gates+1)]:
            coeff[k,index]=1
            if circuit_depth(supports,coeff[k])>max_depth:coeff[k,index]=0
    initial=coeff.copy();ft=torch.tensor(1.-2*features,dtype=torch.float64,device=device)
    phase=torch.tensor(1.-2*((coeff.astype(np.int64)@features)%2),dtype=torch.float64,device=device)
    off=sparse.csr_matrix(operator).copy();off.setdiag(0);off.eliminate_zeros()
    small=len(np.unique(off.nonzero()[0]))<operator.shape[0]//2
    objective=(SmallOffdiagonalPhaseObjective if small else SparsePhaseObjective)(operator,alpha,starts,device)
    with torch.no_grad():
        current=objective(phase);best_values=current.clone();best=coeff.copy()
        for step in range(flips):
            proposal=rng.integers(len(supports),size=starts);candidate=coeff.copy();candidate[np.arange(starts),proposal]^=1
            allowed=np.array([c.sum()<=max_gates and circuit_depth(supports,c)<=max_depth for c in candidate])
            proposed=phase*ft[torch.tensor(proposal,device=device)];new=objective(proposed)
            temperature=.04*max(0,1-2*step/max(flips,1));diff=(new-current).cpu().numpy()
            accept=allowed&((diff<0)|(rng.random(starts)<np.exp(np.minimum(0,-diff/max(temperature,1e-15)))))
            accept_t=torch.tensor(accept,device=device);current=torch.where(accept_t,new,current)
            phase=torch.where(accept_t[:,None],proposed,phase);coeff[accept]=candidate[accept]
            improve=(current<best_values-1e-13).cpu().numpy();best_values=torch.minimum(current,best_values);best[improve]=coeff[improve]
    values=best_values.cpu().numpy();index=int(np.argmin(values))
    return {'kind':'diagonal_circuit','objective':'Operator_Walshness_only','alpha':alpha,'seed':seed,
        'gate_supports':supports,'gate_family':'complement_paired_diagonal' if symmetric else 'Z_CZ_CCZ',
        'gatewise_Z2_symmetric':symmetric,'initial_coefficients':initial,'coefficients_restarts':best,
        'coefficients_best':best[index],'W_restarts':values,'W_best':float(values[index]),
        'gate_count':int(best[index].sum()),'circuit_depth':circuit_depth(supports,best[index]),
        'max_gates':max_gates,'max_depth':max_depth,'flips_per_start':flips,'evaluations':starts*(flips+1),
        'seconds':time.monotonic()-begin,'pure_projector_objective':False,'certified_global':False,
        'backend':'exact_full_diagonal_plus_small_offdiagonal_support' if small else 'block_sparse_gauged_O_times_product_A'}
