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
def test_small_offdiagonal_decomposition_value_gradient_and_phase():
    import numpy as np
    import torch
    from scipy import sparse
    from operator_walshness.signrule.sparse_blind import (SmallOffdiagonalMoment,SmallOffdiagonalPhaseObjective,
        sparse_operator,sparse_hermitian_moment,SparsePhaseObjective)
    rng=np.random.default_rng(13);n=5;d=1<<n
    operator=np.diag(rng.uniform(.1,1,d)).astype(complex)
    ids=[2,7,16,30];block=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));block=block+block.conj().T
    operator[np.ix_(ids,ids)]+=block
    op=sparse.csr_matrix(operator);axes=torch.tensor(rng.normal(size=(3,n,3)),dtype=torch.float64,requires_grad=True)
    normalized=axes/axes.norm(dim=-1,keepdim=True)
    actual=SmallOffdiagonalMoment(op,'cpu')(normalized,.7)
    expected=sparse_hermitian_moment(sparse_operator(op,'cpu'),normalized,.7)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-12)
    ga=torch.autograd.grad(actual.sum(),axes,retain_graph=True)[0]
    ge=torch.autograd.grad(expected.sum(),axes)[0]
    torch.testing.assert_close(ga,ge,atol=1e-10,rtol=1e-10)
    signs=torch.tensor(rng.choice([-1.,1.],(3,d)))
    torch.testing.assert_close(SmallOffdiagonalPhaseObjective(op,.7,3,'cpu')(signs),
        SparsePhaseObjective(op,.7,3,'cpu')(signs),atol=1e-12,rtol=1e-12)
def test_supported_pure_value_gradient_and_circuit_reconstruction():
    import numpy as np
    import torch
    from operator_walshness.signrule.blind import supported_pure_moment,pure_moment_torch,optimize_circuit,circuit_phase
    from operator_walshness.walsh import state_walshness
    rng=np.random.default_rng(9);psi=np.zeros(64,complex);ids=np.array([1,12,43,56]);psi[ids]=rng.normal(size=4)+1j*rng.normal(size=4)
    raw=torch.tensor(rng.normal(size=(2,6,3)),dtype=torch.float64,requires_grad=True);axes=raw/raw.norm(dim=-1,keepdim=True)
    actual=supported_pure_moment(torch.tensor(psi[ids]),torch.tensor(ids),axes,2.)
    expected=pure_moment_torch(torch.tensor(psi),axes,2.)
    torch.testing.assert_close(actual,expected,atol=1e-8,rtol=1e-12)
    a=torch.autograd.grad(actual.sum(),raw,retain_graph=True)[0];b=torch.autograd.grad(expected.sum(),raw)[0]
    torch.testing.assert_close(a,b,atol=1e-8,rtol=1e-10)
    result=optimize_circuit(psi,2.,[(i,) for i in range(6)]+[(i,i+1) for i in range(5)],starts=3,flips=20,pure=True)
    assert abs(result['W_best']-2*state_walshness(psi*circuit_phase(6,result),2.))<1e-10
