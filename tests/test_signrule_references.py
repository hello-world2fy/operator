import numpy as np
import pytest
from operator_walshness.signrule.geometry import triangular
from operator_walshness.signrule.huse_elser import reference_features,trial_wavefunction,fit_circular_phase
from operator_walshness.signrule.diagnostics import phase_entries
from operator_walshness.signrule.blind import alpha0_from_hs_rows
from operator_walshness.walsh import intrinsic_alpha0,walshness,rotate_operator


def test_huse_elser_cubic_rotation_symmetry():
    g=triangular(3,3);_,_,cubic,meta=reference_features(g)
    mapping=[(-y)%3+3*((x+y)%3) for y in range(3) for x in range(3)]
    states=np.arange(512);rotated=np.zeros(512,dtype=int)
    for old,new in enumerate(mapping):rotated|=((states>>old)&1)<<new
    assert np.allclose(cubic[rotated],-cubic)
    assert meta['nonzero_monomials_after_periodic_aliasing']>0
    psi,_=trial_wavefunction(g,n_down=4)
    assert np.linalg.norm(psi)==pytest.approx(1)
    assert np.all(psi[np.bitwise_count(states)!=4]==0)


def test_continuous_phase_fit_retains_U1_phases():
    rng=np.random.default_rng(2);features=rng.normal(size=(24,2))
    phase=np.exp(1j*(features@np.array([.31,-.6])))
    operator=phase[:,None]*phase[None,:].conj()
    prediction,fit=fit_circular_phase(phase_entries(operator),features,starts=3,maxiter=100)
    assert fit['test']['circular_loss']<1e-10
    assert np.max(abs(prediction.imag))>.1


def test_hs_row_alpha0_and_symmetric_single_site_W_invariance():
    rng=np.random.default_rng(55);a=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16));a=a+a.conj().T
    value,axes=alpha0_from_hs_rows(a);expected,directions=intrinsic_alpha0(a)
    assert value==pytest.approx(expected,abs=1e-12)
    assert np.allclose(axes,directions,atol=1e-12)
    x=np.array([[0,1],[1,0]]);unitaries=[np.cos(t)*np.eye(2)-1j*np.sin(t)*x for t in [.3,.5,.2,.7]]
    rotated=rotate_operator(a,unitaries)
    for alpha in [.5,1,2]:assert walshness(rotated,alpha)==pytest.approx(walshness(a,alpha),abs=1e-12)


def test_random_local_gauge_preserves_spectrum_but_invalidates_raw_sign_labels():
    from operator_walshness.signrule.models import paramagnet_gibbs,boolean_phase
    from operator_walshness.signrule.diagnostics import score_phase
    rng=np.random.default_rng(711);phase=boolean_phase(4,[(0,1),(1,2),(2,3)])
    rho=paramagnet_gibbs(4,.8,phase);unitaries=[]
    for _ in range(4):
        raw=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2));u,_=np.linalg.qr(raw);unitaries.append(u)
    changed=rotate_operator(rho,unitaries)
    restored=rotate_operator(changed,[u.conj().T for u in unitaries])
    assert np.allclose(restored,rho,atol=1e-14)
    assert np.allclose(np.linalg.eigvalsh(changed),np.linalg.eigvalsh(rho),atol=1e-14)
    assert score_phase(phase_entries(rho),phase)['circular_loss']<1e-25
    assert score_phase(phase_entries(changed),phase)['circular_loss']>.05
    for alpha in [.5,1,2]:assert walshness(restored,alpha)==pytest.approx(walshness(rho,alpha),abs=1e-12)
