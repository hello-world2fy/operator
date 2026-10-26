import numpy as np
import torch
import pytest
from scipy import sparse
from operator_walshness.optimize import batched_moment
from operator_walshness.walsh import walshness_from_directions,walshness,intrinsic_alpha0
from operator_walshness.signrule.sparse_blind import (sparse_operator,sparse_hermitian_moment,optimize_sparse_local,
    SparsePhaseObjective,optimize_sparse_circuit)
from operator_walshness.signrule.blind import circuit_phase


def test_sparse_moment_and_gradient_match_dense_physical_objective():
    rng=np.random.default_rng(42);a=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16));a[a.real<.2]=0;op=a+a.conj().T
    axes=rng.normal(size=(3,4,3));axes/=np.linalg.norm(axes,axis=-1,keepdims=True)
    for alpha in [.5,1.,2.]:
        x=torch.tensor(axes,requires_grad=True);y=torch.tensor(axes,requires_grad=True)
        m=sparse_hermitian_moment(sparse_operator(op,'cpu'),x,alpha)
        reference=batched_moment(torch.tensor(op),y,alpha)
        assert np.allclose(m.detach(),reference.detach(),rtol=1e-12)
        torch.log(m).sum().backward();torch.log(reference).sum().backward()
        assert np.allclose(x.grad,y.grad,atol=1e-11,rtol=1e-11)
    fit=optimize_sparse_local(sparse.csr_matrix(op),1,starts=4,steps=5,polish=2)
    assert fit['W_best']==pytest.approx(walshness_from_directions(op,fit['axes_best'],1),abs=1e-11)
    assert np.allclose(fit['alpha0_axes'],intrinsic_alpha0(op)[1],atol=1e-11)


@pytest.mark.parametrize('complex_input',[False,True])
def test_sparse_phase_objective_matches_full_fourier_and_candidate_reconstruction(complex_input):
    rng=np.random.default_rng(12);a=rng.normal(size=(16,16))
    if complex_input:a=a+1j*rng.normal(size=a.shape)
    a[abs(a)<.6]=0;op=a+a.conj().T
    phases=1.-2*rng.integers(0,2,(4,16))
    objective=SparsePhaseObjective(sparse.csr_matrix(op),1.2,4,'cpu')
    values=objective(torch.tensor(phases)).numpy()
    expected=[walshness(op*p[:,None]*p[None,:],1.2) for p in phases]
    assert np.allclose(values,expected,atol=1e-11)
    fit=optimize_sparse_circuit(sparse.csr_matrix(op),1.2,[(0,),(1,),(2,),(3,),(0,1)],starts=4,flips=10)
    p=circuit_phase(4,fit)
    assert fit['W_best']==pytest.approx(walshness(op*p[:,None]*p[None,:],1.2),abs=1e-11)
