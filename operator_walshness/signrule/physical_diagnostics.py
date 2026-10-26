"""Independent physical observables and phase-problem diagnostics after freezing."""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh,svdvals
from scipy.special import logsumexp
from scipy import sparse
from .diagnostics import phase_entries,score_phase


def worldline_phase_severity(hamiltonian,beta,n,eigenvalues=None):
    """Exact continuous-time worldline average phase, not a Gibbs-entry metric.

    Off-diagonal H_abs=-abs(H_st); diagonal H_abs=H_ss. This includes complex
    bases as a phase-quenched diagnostic and coincides with average sign for
    real bases. The physical partition function uses the original spectrum.
    """
    h=np.asarray(hamiltonian);e=eigh(h,eigvals_only=True,check_finite=False) if eigenvalues is None else eigenvalues
    quenched=-abs(h);np.fill_diagonal(quenched,np.diag(h).real)
    ea=eigh(quenched,eigvals_only=True,check_finite=False)
    logz=float(logsumexp(-beta*e));logza=float(logsumexp(-beta*ea));log_phase=logz-logza
    if log_phase>1e-8:raise ArithmeticError('phase-quenched partition bound violated')
    return {'logZ':logz,'logZ_abs':logza,'log_average_phase':log_phase,
            'average_phase':float(np.exp(log_phase)),'gamma':float(-log_phase/(beta*n)) if beta else 0.,
            'complex_basis':bool(np.max(abs(h.imag))>1e-10),'Monte_Carlo_sampling':False}


def tj_binary_encoding(digits):
    """Two local diagonal signs/site: hole indicator and down-spin indicator.

    This is only an endpoint gauge feature encoding, NEVER a qubit Walsh metric.
    Exhaustive linear gauge optimization then uses a Boolean character transform.
    """
    n=digits.shape[1]
    return (digits==2).astype(np.int64)@(1<<np.arange(n))+(digits==0).astype(np.int64)@(1<<np.arange(n,2*n))


def constrained_linear_fit(entries,codes,split='train'):
    from ..walsh import fwht
    selected=~entries['test'] if split=='train' else entries['test'] if split=='test' else np.ones(len(entries['row']),bool)
    nbits=int(np.max(codes)).bit_length()
    xor=codes[entries['row'][selected]]^codes[entries['col'][selected]]
    correlation=np.bincount(xor,weights=entries['weight'][selected]*entries['phase'][selected].real,minlength=1<<nbits)
    coefficients=int(np.argmax(fwht(correlation)))
    phase=1.-2*(np.bitwise_count(codes.astype(np.uint64)&coefficients)%2).astype(float)
    return phase,coefficients


def constrained_features(digits,edges,level=2):
    features=[];labels=[]
    for i in range(digits.shape[1]):
        for state in [0,2]:features.append(digits[:,i]==state);labels.append([(i,state)])
    if level>=2:
        for i,j in edges:
            for a in [0,2]:
                for b in [0,2]:
                    f=(digits[:,i]==a)&(digits[:,j]==b)
                    if f.any():features.append(f);labels.append([(i,a),(j,b)])
    return np.asarray(features,dtype=np.int8),labels


def fit_constrained_boolean(entries,features,*,max_gates,starts=16,flips=2000,seed=20261027):
    """Post-hoc supervised gauge fit with exactly the stated proposal/gate cap."""
    train=~entries['test'];a,b=entries['row'][train],entries['col'][train]
    c=entries['weight'][train]*entries['phase'][train].real
    signatures=1.-2*(features[:,a]^features[:,b]).astype(float)
    rng=np.random.default_rng(seed);best=None;objectives=[]
    for restart in range(starts):
        coef=np.zeros(len(features),dtype=np.int8)
        if restart:coef[rng.choice(len(features),rng.integers(max_gates+1),replace=False)]=1
        predicted=np.prod(signatures[coef.astype(bool)],axis=0) if coef.any() else np.ones(len(c))
        value=float(c@predicted)
        for _ in range(flips):
            k=int(rng.integers(len(features)))
            if not coef[k] and coef.sum()>=max_gates:continue
            candidate=float(c@(predicted*signatures[k]))
            if candidate>value+1e-15:coef[k]^=1;predicted*=signatures[k];value=candidate
        objectives.append(value)
        if best is None or value>best[0]:best=(value,coef.copy())
    phase=1.-2*((best[1].astype(np.int64)@features)%2)
    return phase,{'coefficients':best[1],'training_objectives':objectives,'max_gates':max_gates,
                 'starts':starts,'flips_per_start':flips,'test':score_phase(entries,phase)}


def spin_hole_correlations(rho,digits):
    diagonal=np.diag(rho).real;hole=(digits==0).astype(float)
    sz=((digits==1).astype(float)-(digits==2))/2
    h=diagonal@hole;s=diagonal@sz
    raw=(hole.T*diagonal)@sz
    return {'hole_density':h,'spin_z':s,'hole_spin_raw':raw,'hole_spin_connected':raw-np.outer(h,s),
            'hole_spin_conditional':raw/np.maximum(h[:,None],1e-300)}


def constrained_operator_entropy(rho,digits,cut=None):
    """Exact spatial operator entropy using only present row/column pair labels.

    Do not embed the operator in the full unconstrained 3^n Hilbert space.
    For larger sectors explicitly decline the expensive SVD rather than label
    a matrix-element Shannon entropy as OSEE.
    """
    n=digits.shape[1];cut=cut or n//2
    left=digits[:,:cut]@(3**np.arange(cut));right=digits[:,cut:]@(3**np.arange(n-cut))
    left_pair=(left[:,None]+3**cut*left[None,:]).ravel()
    right_pair=(right[:,None]+3**(n-cut)*right[None,:]).ravel()
    _,li=np.unique(left_pair,return_inverse=True);_,ri=np.unique(right_pair,return_inverse=True)
    shape=(int(li.max())+1,int(ri.max())+1)
    if np.prod(shape)>20_000_000:return {'computed':False,'compressed_shape':shape,'reason':'exact spatial SVD exceeds registered small-system budget'}
    matrix=np.zeros(shape,dtype=rho.dtype)
    # Mapping of a constrained matrix entry to a spatial pair is injective.
    matrix[li,ri]=rho.ravel()
    singular=svdvals(matrix,check_finite=False);weights=singular**2/np.vdot(rho,rho).real
    positive=weights>0
    return {'computed':True,'OSEE':float(-weights[positive]@np.log2(weights[positive])),
            'Schmidt_weights':weights,'compressed_shape':shape,'cut':cut}


def qutrit_full_spectrum(rho,digits):
    """Exact feasible small-system Fourier spectrum, named separately from Walsh."""
    n=digits.shape[1]
    if n>6:raise ValueError('full qutrit Fourier matrix limited to n<=6; use kernel moments otherwise')
    d=3**n;codes=digits@(3**np.arange(n));full=np.zeros((d,d),complex)
    full[np.ix_(codes,codes)]=rho
    tensor=full.reshape([3]*(2*n))
    # Unitary Fourier on row and inverse Fourier on column; degree ignores +/-1.
    for axis in range(n):tensor=np.fft.fft(tensor,axis=axis,norm='ortho')
    for axis in range(n,2*n):tensor=np.fft.ifft(tensor,axis=axis,norm='ortho')
    q=tensor.reshape(d,d);hw=np.array([sum(x//(3**i)%3!=0 for i in range(n)) for x in range(d)])
    weights=abs(q)**2;norm=weights.sum()
    spectrum=np.bincount((hw[:,None]+hw[None,:]).ravel(),weights=weights.ravel()/norm,minlength=2*n+1)
    return spectrum


def sparse_operator_entropy(operator,n,cut=None):
    """Exact OSEE via connected blocks of the sparse spatial Gram matrix."""
    from scipy.sparse.csgraph import connected_components
    cut=cut or n//2;da=1<<cut;db=1<<(n-cut)
    coo=sparse.coo_matrix(operator)
    left=(coo.row%da)*da+coo.col%da;right=(coo.row//da)*db+coo.col//da
    matrix=sparse.coo_matrix((coo.data,(left,right)),shape=(da*da,db*db)).tocsr()
    gram=(matrix@matrix.conj().T).tocsr();gram.eliminate_zeros()
    _,labels=connected_components(abs(gram),directed=False)
    eigenvalues=[];sizes=[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label);sizes.append(len(ids))
        eigenvalues.extend(eigh(gram[ids][:,ids].toarray(),eigvals_only=True,check_finite=False))
    eigenvalues=np.asarray(eigenvalues);norm=float(np.vdot(coo.data,coo.data).real)
    if eigenvalues.min() < -1e-10*norm:raise ArithmeticError('spatial Gram matrix not positive')
    weights=np.maximum(eigenvalues/norm,0);positive=weights>0
    return {'OSEE':float(-weights[positive]@np.log2(weights[positive])),
            'Schmidt_weights':np.sort(weights)[::-1],'Gram_block_sizes':sizes,
            'Schmidt_weight_sum':float(weights.sum()),'cut':cut,'exact_sparse_spatial_reindex':True}
