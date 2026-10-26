"""Post-freeze scoring. This module must never be imported by blind optimizers."""
from __future__ import annotations

from collections import Counter, deque
import numpy as np
from scipy import sparse
from ..walsh import fwht


def phase_entries(operator,relative_threshold=1e-8,seed=913,holdout=.3):
    """Undirected, off-diagonal entry split: Hermitian partners never leak across splits."""
    operator=np.asarray(operator);d=len(operator)
    a,b=np.triu_indices(d,1);z=operator[a,b]
    threshold=relative_threshold*abs(operator).max(initial=0.)
    good=abs(z)>max(threshold,np.finfo(float).tiny)
    a,b,z=a[good],b[good],z[good]
    # Deterministic pair split, independent of numerical values and threshold.
    key=(a.astype(np.uint64)*np.uint64(0x9E3779B185EBCA87)) ^ b.astype(np.uint64) ^ np.uint64(seed)
    key^=key>>np.uint64(30);key*=np.uint64(0xBF58476D1CE4E5B9)
    key^=key>>np.uint64(27);key*=np.uint64(0x94D049BB133111EB)
    key^=key>>np.uint64(31)
    test=(key%np.uint64(1000000))<int(holdout*1000000)
    return {'row':a,'col':b,'phase':z/abs(z),'weight':2*abs(z)**2,'test':test,
            'threshold':threshold,'relative_threshold':relative_threshold,
            'retained_offdiag_mass':float(2*np.sum(abs(z)**2)),
            'total_mass':float(np.vdot(operator,operator).real),
            'diagonal_mass':float(np.sum(abs(np.diag(operator))**2))}


def score_phase(entries,phase,split='test'):
    selection=entries['test'] if split=='test' else (~entries['test'] if split=='train' else np.ones(len(entries['row']),bool))
    a,b=entries['row'][selection],entries['col'][selection]
    w=entries['weight'][selection];actual=entries['phase'][selection]
    if not len(w) or w.sum()==0:return {'circular_loss':None,'weighted_accuracy':None,'count':0,'mass':0.}
    prediction=phase[a]*np.conj(phase[b])
    loss=abs(actual-prediction)**2/4
    mean=float(w@loss/w.sum())
    # Accuracy is phase agreement within tolerance, separate from smooth loss.
    acc=float(w@(abs(actual-prediction)<1e-7)/w.sum())
    return {'circular_loss':mean,'weighted_accuracy':acc,'unweighted_accuracy':float(np.mean(abs(actual-prediction)<1e-7)),
            'phase_locking':float(abs(w@(actual*np.conj(prediction)))/w.sum()),
            'count':int(len(w)),'mass':float(w.sum())}


def best_linear_phase(entries,n,split='train'):
    """Exhaustive exact optimum of all 2^n Boolean linear endpoint rules via FWHT."""
    selection=~entries['test'] if split=='train' else (entries['test'] if split=='test' else np.ones(len(entries['row']),bool))
    a,b=entries['row'][selection],entries['col'][selection]
    corr=np.bincount(a^b,weights=entries['weight'][selection]*entries['phase'][selection].real,minlength=1<<n)
    scores=fwht(corr)
    coefficient=int(np.argmax(scores))
    phase=1.-2*(np.bitwise_count(np.arange(1<<n,dtype=np.uint64)&coefficient)%2).astype(float)
    return phase,coefficient


def phase_features(n,supports):
    states=np.arange(1<<n,dtype=np.int64)
    return np.array([((states&sum(1<<int(i) for i in support))==sum(1<<int(i) for i in support))
                     for support in supports],dtype=np.int8)


def fit_boolean_phase(entries,n,supports,*,starts=16,flips=2000,seed=0,max_gates=None):
    """Supervised post-hoc fit, distinct from Walsh optimization. Training pairs only."""
    rng=np.random.default_rng(seed);features=phase_features(n,supports)
    train=~entries['test'];a,b=entries['row'][train],entries['col'][train]
    c=entries['weight'][train]*entries['phase'][train].real
    pair_signs=1.-2*(features[:,a]^features[:,b]).astype(float)
    best=None;traces=[]
    for start in range(starts):
        coefficients=np.zeros(len(supports),dtype=np.int8)
        if start:
            chosen=rng.choice(len(supports),size=rng.integers(0,min(len(supports),max_gates or len(supports))+1),replace=False)
            coefficients[chosen]=1
        state_sign=np.prod(pair_signs[coefficients.astype(bool)],axis=0) if coefficients.any() else np.ones(len(c))
        value=float(c@state_sign)
        for _ in range(flips):
            k=int(rng.integers(len(supports)))
            if max_gates is not None and not coefficients[k] and coefficients.sum()>=max_gates:continue
            candidate=float(c@(state_sign*pair_signs[k]))
            if candidate>value+1e-15:
                state_sign*=pair_signs[k];coefficients[k]^=1;value=candidate
        traces.append(value)
        if best is None or value>best[0]:best=(value,coefficients.copy())
    coefficients=best[1]
    parity=(coefficients@features)%2
    return 1.-2*parity,{'coefficients':coefficients,'supports':supports,'training_objective':best[0],
                        'restart_objectives':traces,'max_gates':max_gates}


def paired_improvement_ci(entries,reference,candidate,seed=0,draws=200):
    """Pair bootstrap of the held-out weighted error difference (reference-candidate)."""
    sel=entries['test'];a,b=entries['row'][sel],entries['col'][sel];w=entries['weight'][sel]
    actual=entries['phase'][sel]
    if len(w)<2:return {'improvement':None,'ci95':None}
    delta=(abs(actual-reference[a]*reference[b].conj())**2-abs(actual-candidate[a]*candidate[b].conj())**2)/4
    mean=float(w@delta/w.sum());rng=np.random.default_rng(seed)
    # Poisson bootstrap over independent undirected entries, all entries retained.
    estimates=[]
    for _ in range(draws):
        wb=w*rng.poisson(1,len(w));total=wb.sum()
        if total:estimates.append(float(wb@delta/total))
    return {'improvement':mean,'ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'sampling_unit':'undirected matrix entry; finite-system descriptive interval'}


def support_balance(operator,relative_threshold=1e-8,phase_tolerance=1e-7,witness_limit=200):
    """Check ALL reliable support edges against a spanning forest gauge.

    A negative non-tree edge closes an obstructed fundamental cycle. Unlike a
    triangle test, this detects long cycles in sparse configuration graphs.
    Complex U(1) holonomy uses the same forest; diagonal entries are not edges.
    """
    if sparse.issparse(operator):
        a=operator.tocsr().astype(complex).copy();scale=abs(a.data).max(initial=0.)
        a.data[abs(a.data)<=relative_threshold*scale]=0;a.setdiag(0);a.eliminate_zeros()
    else:
        dense=np.asarray(operator);scale=abs(dense).max(initial=0.)
        a=sparse.csr_matrix(np.where(abs(dense)>relative_threshold*scale,dense,0)).astype(complex)
        a.setdiag(0);a.eliminate_zeros()
    d=a.shape[0];gauge=np.ones(d,complex);visited=np.zeros(d,bool)
    parent=np.full(d,-1,int);depth=np.zeros(d,int);components=0
    for root in range(d):
        if visited[root]:continue
        components+=1;visited[root]=True;q=deque([root])
        while q:
            s=q.popleft()
            for k in range(a.indptr[s],a.indptr[s+1]):
                t=a.indices[k]
                if visited[t]:continue
                gauge[t]=gauge[s]*np.conj(a.data[k]/abs(a.data[k]));visited[t]=True
                parent[t]=s;depth[t]=depth[s]+1;q.append(t)
    upper=sparse.triu(a,k=1).tocoo()
    residual=abs(upper.data/abs(upper.data)-gauge[upper.row]*gauge[upper.col].conj())
    bad=np.flatnonzero(residual>phase_tolerance);lengths=[];witnesses=[]
    for k in bad[:witness_limit]:
        s,t=int(upper.row[k]),int(upper.col[k]);left=[s];right=[t]
        while s!=t:
            if depth[s]>=depth[t]:s=parent[s];left.append(int(s))
            else:t=parent[t];right.append(int(t))
        path=left+right[-2::-1]
        lengths.append(len(path));witnesses.append(path)
    return {'balanced':bool(len(bad)==0),'components':components,'support_edges_checked':int(len(upper.data)),
            'obstructed_fundamental_edges':int(len(bad)),
            'obstructed_fraction':float(len(bad)/max(len(upper.data),1)),
            'max_holonomy_residual':float(residual.max(initial=0.)),
            'sampled_negative_cycle_lengths':dict(Counter(lengths)),
            'witness_cycles':witnesses,'all_support_edges_checked':True,'gauge':gauge}


def path_phase_samples(h,h_sigma,hops,gauge,*,lengths=(4,8,12,20),samples=2000,seed=0):
    """Actual closed configuration paths (hops AND spin exchange), no endpoint inference.

    Random walks sample off-diagonal expansion events, with unsuccessful return
    walks retained in counts. Report loop parity and the exact product of -H
    signs; the sigma model must have trivial holonomy in this bipartite gauge.
    This is not importance sampling of a QMC partition function.
    """
    hm=sparse.diags(gauge)@h@sparse.diags(gauge)
    sm=sparse.diags(gauge)@h_sigma@sparse.diags(gauge)
    hm=hm.tocsr();sm=sm.tocsr();down={(int(a),int(b)):int(d) for a,b,d,_,_ in hops}
    adjacency=[]
    for s in range(h.shape[0]):
        neighbors=hm.indices[hm.indptr[s]:hm.indptr[s+1]]
        adjacency.append(neighbors[neighbors!=s])
    rng=np.random.default_rng(seed);out=[];records=[]
    for length in lengths:
        closed=negative=errors=off_errors=0
        for sample in range(samples):
            start=int(rng.integers(h.shape[0]));s=start;sign=1.;sign_off=1.;parity=0;trajectory=[s]
            for _ in range(length):
                if not len(adjacency[s]):break
                t=int(rng.choice(adjacency[s]));sign*=np.sign(-hm[t,s].real);sign_off*=np.sign(-sm[t,s].real)
                parity^=down.get((s,t),0);s=t;trajectory.append(s)
            if len(trajectory)!=length+1 or s!=start:continue
            closed+=1;negative+=sign<0;errors+=sign!=(-1.)**parity;off_errors+=sign_off!=1
            records.append({'length':length,'start':start,'phase_string_parity':parity,
                            'path_sign':int(sign),'sigma_path_sign':int(sign_off),'trajectory':trajectory})
        out.append({'length':length,'attempts':samples,'closed':closed,'negative':int(negative),
                    'phase_string_violations':int(errors),'sigma_nontrivial':int(off_errors)})
    return out,records


def strange_correlator(psi,n):
    """<+|Zi Zj|psi>/<+|psi>, with denominator reliability explicitly reported."""
    states=np.arange(1<<n);z=1.-2*((states[None,:]>>np.arange(n)[:,None])&1)
    denominator=np.sum(psi)
    if abs(denominator)<1e-10*np.sum(abs(psi)):
        return {'defined':False,'overlap':complex(denominator/np.sqrt(len(psi))),
                'reason':'reference overlap cancels on this finite torus'}
    return {'defined':True,'overlap':complex(denominator/np.sqrt(len(psi))),
            'correlation':(z*psi)@z.T/denominator}
