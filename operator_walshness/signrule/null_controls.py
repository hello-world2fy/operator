"""Matched-spectrum null ensembles, with explicit symmetry-preserving sectors."""
from __future__ import annotations
import numpy as np
from scipy.linalg import qr,block_diag


def involution_basis(permutation):
    permutation=np.asarray(permutation);d=len(permutation)
    assert np.array_equal(permutation[permutation],np.arange(d))
    plus=[];minus=[]
    for i,j in enumerate(permutation):
        if i>j:continue
        v=np.zeros(d)
        if i==j:v[i]=1;plus.append(v)
        else:
            v[i]=v[j]=1/np.sqrt(2);plus.append(v)
            w=v.copy();w[j]*=-1;minus.append(w)
    return np.column_stack(plus+minus),len(plus)


def haar_matrix(d,rng,real=False):
    z=rng.normal(size=(d,d))
    if not real:z=z+1j*rng.normal(size=(d,d))
    q,r=qr(z,mode='economic',check_finite=False);diagonal=np.diag(r)
    return q*(diagonal/abs(diagonal)).conj()[None,:]


def same_spectrum_null(rho,family,seed,permutation):
    rng=np.random.default_rng(seed)
    if family=='sector_Haar':
        basis,split=involution_basis(permutation)
        if np.max(abs(rho[np.ix_(permutation,permutation)]-rho))>1e-10:
            raise ValueError('source does not preserve the registered involution')
        u=basis@block_diag(haar_matrix(split,rng),haar_matrix(len(rho)-split,rng))@basis.T
    elif family in ['complex_Haar','real_orthogonal_Haar']:
        u=haar_matrix(len(rho),rng,real=family=='real_orthogonal_Haar')
    else:raise ValueError(family)
    rotated=u.conj().T@rho@u
    residual={'unitarity_max':float(abs(u.conj().T@u-np.eye(len(u))).max()),
        'Hermiticity_max':float(abs(rotated-rotated.conj().T).max()),
        'trace_difference':float(abs(np.trace(rotated)-np.trace(rho))),
        'purity_difference':float(abs(np.vdot(rotated,rotated)-np.vdot(rho,rho))),
        'involution_commutator_max':float(abs(rotated[np.ix_(permutation,permutation)]-rotated).max())}
    return np.real_if_close(rotated),residual
