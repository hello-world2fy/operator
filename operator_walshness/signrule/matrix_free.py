"""Sparse propagator and ground-sector diagnostics with explicit sampling limits."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh,expm_multiply
from .diagnostics import phase_entries


def verified_ground(h,seed=20261027):
    """Use independent random Krylov starts, checking residuals and ground agreement."""
    solutions=[];n=len(h.indptr)-1
    for repeat in range(2):
        rng=np.random.default_rng(seed+repeat);v0=rng.normal(size=n)
        if np.iscomplexobj(h.data):v0=v0+1j*rng.normal(size=n)
        energies,vectors=eigsh(h,k=min(8,n-2),which='SA',v0=v0,tol=1e-11,maxiter=10000)
        order=np.argsort(energies);energies=energies[order];vectors=vectors[:,order]
        residual=np.linalg.norm(h@vectors-vectors*energies,axis=0)
        if residual.max()>1e-8:raise ArithmeticError('Krylov eigenpair residual')
        solutions.append((energies,vectors,residual))
    if abs(solutions[0][0][0]-solutions[1][0][0])>1e-8:raise ArithmeticError('independent Krylov ground mismatch')
    energies,vectors,residual=solutions[0];ground=abs(energies-energies[0])<1e-8
    if ground.all():raise ArithmeticError('increase eigenpair budget to resolve ground degeneracy')
    return energies,vectors,{'ground_energy':float(energies[0]),'resolved_ground_degeneracy':int(ground.sum()),
        'first_excitation_gap':float(energies[~ground][0]-energies[0]),'max_eigenpair_residual':float(residual.max()),
        'two_start_ground_difference':float(abs(energies[0]-solutions[1][0][0])),
        'random_Krylov_starts':2,'v0_is_not_a_symmetry_selected_constant':True}


def propagator_columns(h,beta,columns,ground_energy):
    rhs=np.zeros((h.shape[0],len(columns)));rhs[columns,np.arange(len(columns))]=1
    shifted=h-ground_energy*sparse.eye(h.shape[0],format='csr')
    return expm_multiply(-beta*shifted,rhs,traceA=-beta*shifted.diagonal().sum())


def sampled_column_entries(values,columns,cutoff=1e-8):
    """Deduplicate Hermitian partners; retain exactly the observed pairs, no inference."""
    d=values.shape[0];columns=np.asarray(columns)
    rows=np.repeat(np.arange(d),len(columns));cols=np.tile(columns,d);z=values.ravel()
    a=np.minimum(rows,cols);b=np.maximum(rows,cols);z=np.where(rows<=cols,z,z.conj())
    key=a*d+b;order=np.argsort(key,kind='stable');key=key[order]
    keep=np.r_[True,key[1:]!=key[:-1]];order=order[keep]
    a,b,z=a[order],b[order],z[order]
    upper=sparse.coo_matrix((z,(a,b)),shape=(d,d)).tocsr()
    observed=upper+upper.conj().T-sparse.diags(upper.diagonal())
    entries=phase_entries(observed,cutoff)
    entries['coverage']='sampled columns only; no complete Gibbs support or purity claim'
    entries['relative_threshold_reference']='maximum absolute observed matrix element'
    return entries


def stochastic_thermal_diagonal(h,beta,ground_energy,*,probes=64,seed=20261027):
    """Positive Hutchinson diagonal estimate from exp(-beta H/2) Rademacher probes."""
    shifted=h-ground_energy*sparse.eye(h.shape[0],format='csr')
    z=np.random.default_rng(seed).choice([-1.,1.],size=(h.shape[0],probes))
    y=expm_multiply(-beta/2*shifted,z,traceA=-beta/2*shifted.diagonal().sum())
    squared=abs(y)**2;traces=squared.sum(0);mean=float(traces.mean())
    diagonal=squared.mean(1)/mean
    return diagonal,{'trace_shifted_estimate':mean,'trace_standard_error':float(traces.std(ddof=1)/np.sqrt(probes)),
        'trace_relative_standard_error':float(traces.std(ddof=1)/np.sqrt(probes)/mean),
        'probes':probes,'seed':seed,'normalized_ratio_estimator':True,'exact_density_diagonal':False},squared
