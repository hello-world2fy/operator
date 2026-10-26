"""Geometric rule transfer, fixed without accessing target operator entries."""
from __future__ import annotations
import numpy as np
from .blind import circuit_depth,gate_features


def triangular_catalog(lx,ly,periodic=True):
    def site(x,y):
        if not periodic and not (0<=x<lx and 0<=y<ly):return None
        return (x%lx)+lx*(y%ly)
    out=[]
    for y in range(ly):
        for x in range(lx):
            out.append((x,y,'Z',(site(x,y),)))
            for kind,(dx,dy) in enumerate([(1,0),(0,1),(1,-1)]):
                target=site(x+dx,y+dy)
                if target is not None:out.append((x,y,f'edge{kind}',tuple(sorted((site(x,y),target)))))
            if periodic or (x+1<lx and y+1<ly):
                out.append((x,y,'triangle0',tuple(sorted((site(x,y),site(x+1,y),site(x,y+1))))))
                out.append((x,y,'triangle1',tuple(sorted((site(x+1,y+1),site(x,y+1),site(x+1,y))))))
    if len({t[3] for t in out})!=len(out):raise ValueError('catalog aliases supports')
    return out


def transfer_triangular_circuit(result,source_shape,target_shape,*,target_periodic=True):
    """Repeat source cell coefficients modulo source lengths, using target-local
    bonds/faces (including the target wrap). Enforce a site-proportional gate
    cap by a fixed geometric ordering, never by a target phase/objective score.
    """
    sx,sy=source_shape;tx,ty=target_shape
    learned={tuple(sorted(s)):int(on) for s,on in zip(result['gate_supports'],result['coefficients_best'])}
    source={(x,y,kind):learned.get(support,0) for x,y,kind,support in triangular_catalog(sx,sy)}
    target=triangular_catalog(tx,ty,target_periodic)
    supports=[entry[3] for entry in target]
    coefficients=np.array([source[x%sx,y%sy,kind] for x,y,kind,s in target],dtype=np.int8)
    before=int(coefficients.sum());cap=int(np.floor(result['max_gates']*tx*ty/(sx*sy)))
    order=sorted(range(len(target)),key=lambda k:(len(target[k][3]),target[k][1],target[k][0],target[k][2]))
    active=[i for i in order if coefficients[i]]
    for index in active[cap:]:coefficients[index]=0
    depth_cap=int(np.ceil(result['max_depth']*tx*ty/(sx*sy)))
    while circuit_depth(supports,coefficients)>depth_cap:
        last=next(i for i in reversed(order) if coefficients[i]);coefficients[last]=0
    result_target={**result,'gate_supports':supports,'coefficients_best':coefficients,
        'max_gates':cap,'max_depth':depth_cap,'gate_count':int(coefficients.sum()),
        'circuit_depth':circuit_depth(supports,coefficients),
        'transfer':'source anchor coefficients repeated modulo source dimensions; target geometric bonds/faces',
        'untrimmed_gate_count':before,'target_label_or_W_access':False}
    # Symmetric gate_features omits single-site identities; keep coefficients aligned.
    if result['gatewise_Z2_symmetric']:
        keep=[i for i,s in enumerate(supports) if len(s)>1]
        result_target['gate_supports']=[supports[i] for i in keep]
        result_target['coefficients_best']=coefficients[keep]
    return result_target


def exact_grouped_phase_bootstrap(entries,reference,candidate,*,draws=200,seed=0):
    """Exact Poisson entry-bootstrap aggregation for equal-weight Hamming shells.

    For the solvable paramagnet/Levin-Gu pair, |rho_st|^2 depends only on s xor t.
    The reference-candidate Boolean error difference is -1,0,+1. Sums of
    independent Poisson(1) counts within each shell/difference group are exactly
    Poisson(group_size), reducing millions of entries to <=3(n+1) groups.
    """
    sel=entries['test'];a,b=entries['row'][sel],entries['col'][sel];w=entries['weight'][sel];actual=entries['phase'][sel]
    if not len(w):return {'improvement':None,'ci95':None}
    delta=(abs(actual-reference[a]*reference[b].conj())**2-abs(actual-candidate[a]*candidate[b].conj())**2)/4
    rounded=np.rint(delta).astype(int)
    if not np.allclose(delta,rounded,atol=1e-12):raise ValueError('Boolean phase differences required')
    shell=np.bitwise_count((a^b).astype(np.uint64)).astype(int)
    key=shell*3+rounded+1;count=np.bincount(key);sum_w=np.bincount(key,weights=w);sum_delta=np.bincount(key,weights=w*delta)
    nonzero=count>0;weight=sum_w[nonzero]/count[nonzero];difference=sum_delta[nonzero]/sum_w[nonzero]
    average=np.zeros(len(count));average[nonzero]=weight
    if np.max(abs(w-average[key])/np.maximum(w,1e-300))>1e-10:raise ValueError('weights are not equal inside shells')
    draws_counts=np.random.default_rng(seed).poisson(count[nonzero],(draws,nonzero.sum()))
    weighted=draws_counts*weight;estimates=(weighted@difference)/weighted.sum(1)
    return {'improvement':float(w@delta/w.sum()),'ci95':np.quantile(estimates,[.025,.975]).tolist(),
        'method':'exact grouped Poisson bootstrap of undirected entries','groups':int(nonzero.sum()),'entries':len(w)}
