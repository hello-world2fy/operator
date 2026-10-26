"""Post-hoc Huse-Elser references, never imported by blind optimization.

The three-site clusters have both endpoints nearest to the center and next-
nearest to each other. Gamma is invariant under 120-degree rotations and flips
under 60 degrees. See PhysRevResearch.4.L012010 supplemental Eqs. S46-S48:
https://journals.aps.org/prresearch/supplemental/10.1103/PhysRevResearch.4.L012010/supp.pdf
The reference is variational, NOT an exact sign oracle for Heisenberg Gibbs states.
"""
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from .diagnostics import score_phase


def geometric_triplets(g):
    if g.name!='triangular':raise ValueError('triangular geometry required')
    lx,ly=g.shape;periodic=g.boundary=='periodic'
    directions=[(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    triples=[]
    for y in range(ly):
        for x in range(lx):
            center=x+lx*y
            for r in range(6):
                points=[(x+directions[k][0],y+directions[k][1]) for k in [r,(r+2)%6]]
                if not periodic and not all(0<=xx<lx and 0<=yy<ly for xx,yy in points):continue
                ends=[xx%lx+lx*(yy%ly) for xx,yy in points]
                if len(set(ends+[center]))<3:raise ValueError('folded triplet self-intersection')
                triples.append((ends[0],center,ends[1],(-1)**r))
    return triples


def reference_features(g):
    configurations=np.arange(1<<g.n);sz=(1.-2*((configurations[:,None]>>np.arange(g.n))&1))/2
    sub=np.array(g.sublattice)
    three_sublattice=np.stack([sz[:,sub==s].sum(1) for s in range(3)],axis=1)
    triples=geometric_triplets(g)
    cubic=sum((gamma*sz[:,i]*sz[:,j]*sz[:,k] for i,j,k,gamma in triples),start=np.zeros(len(sz)))
    coefficients=Counter()
    for i,j,k,gamma in triples:coefficients[tuple(sorted((i,j,k)))]+=gamma
    meta={'geometric_triplets':len(triples),'nonzero_monomials_after_periodic_aliasing':sum(v!=0 for v in coefficients.values()),
          'cubic_feature_variance':float(np.var(cubic)),
          'triplet_convention':'outer sites at 120 degrees; gamma=(-1)^orientation index',
          'exact_oracle':False}
    return sz,three_sublattice,cubic,meta


def trial_wavefunction(g,cubic_strength=.13,jastrow=1.45,real_part=False,n_down=None):
    sz,sub,cubic,meta=reference_features(g)
    linear=2*np.pi/3*(sub[:,1]-sub[:,2])
    amplitude=np.exp(-jastrow/2*sum((sz[:,a]*sz[:,b] for a,b in g.edges),start=np.zeros(len(sz))))
    phase=linear+cubic_strength*cubic
    psi=amplitude*(np.cos(phase) if real_part else np.exp(1j*phase))
    if n_down is not None:
        psi[np.bitwise_count(np.arange(1<<g.n,dtype=np.uint64))!=n_down]=0
    psi/=np.linalg.norm(psi)
    return psi,meta


def fit_circular_phase(entries,features,*,starts=16,maxiter=300,seed=20261027):
    """Supervised phase fit on training pairs, with continuous U(1) residuals."""
    selected=~entries['test'];a,b=entries['row'][selected],entries['col'][selected]
    delta=features[a]-features[b];w=entries['weight'][selected];actual=entries['phase'][selected]
    if len(w)==0:return np.ones(len(features),complex),{'coefficients':np.zeros(features.shape[1]),'scores':[]}
    w=w/w.sum();rng=np.random.default_rng(seed);best=None;scores=[]
    def objective(parameters):
        phase=np.exp(1j*(delta@parameters));cross=actual.conj()*phase
        value=float(np.dot(w,(1-cross.real)/2))
        gradient=delta.T@(w*cross.imag)/2
        return value,gradient
    for restart in range(starts):
        initial=np.zeros(features.shape[1]) if restart==0 else rng.uniform(-np.pi,np.pi,features.shape[1])
        result=minimize(objective,initial,method='L-BFGS-B',jac=True,options={'maxiter':maxiter,'ftol':1e-12})
        scores.append(float(result.fun))
        if best is None or result.fun<best.fun:best=result
    prediction=np.exp(1j*(features@best.x))
    return prediction,{'coefficients':best.x,'scores':scores,'train':score_phase(entries,prediction,'train'),
                       'test':score_phase(entries,prediction),'converged':bool(best.success),
                       'gradient_norm':float(np.linalg.norm(best.jac)),'parameter_count':features.shape[1]}


def structure_factor_K(rho,g):
    """Full SU2 scalar structure factor sum exp(i K.(ri-rj)) <Si.Sj>/n."""
    n=g.n;states=np.arange(1<<n);z=1.-2*((states[:,None]>>np.arange(n))&1)
    correlation=np.empty((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:correlation[i,j]=.75
            else:
                diag=float(np.diag(rho).real@(z[:,i]*z[:,j]))/4
                opposite=((states>>i)^(states>>j))&1
                exchange=float(np.sum(rho[states,states^(1<<i)^(1<<j)]*opposite).real)/2
                correlation[i,j]=diag+exchange
    phase=np.exp(2j*np.pi/3*np.array(g.sublattice))
    return float(np.einsum('i,ij,j->',phase.conj(),correlation,phase).real/n),correlation


def axis_texture(axes,g):
    axes=np.asarray(axes);sub=np.asarray(g.sublattice)
    means=np.array([axes[sub==s].mean(0) for s in range(3)])
    pairdots=np.array([[np.mean(axes[sub==s]@axes[sub==t].T) for t in range(3)] for s in range(3)])
    return {'sublattice_mean_vectors':means,'mean_vector_lengths':np.linalg.norm(means,axis=1),
            'sublattice_axis_dot_products':pairdots,'ideal_distinct_sublattice_dot':-.5,
            'interpretation':'basis texture only; not an HE sign-rule certificate'}
