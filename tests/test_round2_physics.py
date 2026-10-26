import numpy as np
import pytest
from scipy.linalg import expm
from operator_walshness.round2 import *
from operator_walshness.hamiltonians import build_hamiltonian
from operator_walshness.observables import operator_schmidt
from operator_walshness.walsh import walshness_from_directions


def test_thermal_identity_complex_axes_and_information_bound():
    rng=np.random.default_rng(53);a=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    h=a+a.conj().T;e,v=np.linalg.eigh(h)
    rho,_=gibbs_density(e,v,.4);rho2,_=gibbs_density(e,v,.8)
    axes=rng.normal(size=(3,3));axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    equal,imaginary,x=thermal_walsh_identity(rho,rho2,axes)
    cov=degree_correlations_from_x_matrix(x)['degree_correlation']
    np.testing.assert_allclose(cov,(equal+imaginary)/2,atol=1e-12)
    s,entropy=schmidt(rho);oldp,oldentropy=operator_schmidt(rho)
    np.testing.assert_allclose(s*s,oldp,atol=1e-12)
    assert entropy==pytest.approx(oldentropy)
    assert walsh_mutual_information(x)<=2*entropy+1e-12


@pytest.mark.parametrize('boundary',['open','periodic'])
def test_marshall_gauge_and_exact_partition_functions(boundary):
    n=4;h=j1j2_hamiltonian(n,0,boundary)
    marshall=real_rotate(h,np.zeros(n),np.arange(n)%2)
    m=sign_metrics(marshall,1.)
    assert abs(m['log_average_sign'])<1e-12
    h=j1j2_hamiltonian(n,.5,boundary)
    hu=real_rotate(h,np.arange(n)*.4,np.arange(n)%2)
    m=sign_metrics(hu,1.2)
    ratio=np.trace(expm(-1.2*hu))/np.trace(expm(-1.2*sign_quenched(hu)))
    assert m['average_sign']==pytest.approx(ratio,abs=1e-12)
    gauge=real_rotate(hu,np.zeros(n),np.array([0,0,1,0]))
    assert sign_metrics(gauge,1.2)['gamma']==pytest.approx(m['gamma'],abs=1e-12)
    assert abs(sign_metrics(hu,1e-5)['log_average_sign'])<1e-10


def test_stoquastic_control_and_centered_high_temperature_limit():
    h=build_hamiltonian('mixed_field_ising',4,g=1.05,h=.5)
    assert abs(sign_metrics(h,2.)['log_average_sign'])<1e-12
    e,v=np.linalg.eigh(h);rho,_=gibbs_density(e,v,1e-7)
    delta=rho-np.eye(len(h))/len(h)
    p,_=fixed_basis_spectrum(delta);q,_=fixed_basis_spectrum(h)
    assert abs(walshness_from_spectrum(p,1)-walshness_from_spectrum(q,1))<1e-6


def test_complex_haar_mean_formula_with_sampling():
    rng=np.random.default_rng(123);p=np.array([.65,.25,.07,.03]);purity=float(p@p)
    values=[]
    for _ in range(2000):
        u=haar_unitary(4,rng);rho=(u*p)@u.conj().T
        spec,_=fixed_basis_spectrum(rho)
        values.append(np.exp(walshness_from_spectrum(spec,.5)*.5))
    values=np.array(values);theory=haar_mean_moment(2,purity,.5)
    assert abs(values.mean()-theory)<5*values.std(ddof=1)/np.sqrt(len(values))
    assert haar_mean_moment(2,.25,.5)==pytest.approx(((1+np.exp(1))/2)**2)


def test_optimization_respects_total_evaluation_budget():
    h=j1j2_hamiltonian(4,.5)
    for objective in ('walsh','sign','eta'):
        result=optimize_real_basis(h,objective,starts=2,max_evals=24)
        assert np.max(result['restart_nfev'])<=24
        assert np.isfinite(result['value'])


def test_result_keys_preserve_decimal_parameters(tmp_path):
    for value in (.25,.5):
        save_result(tmp_path,'R3',f'b{value:g}_complex',[{'beta':value}],P=np.array([value]))
    assert len(list((tmp_path/'R3').glob('*.json')))==2
    assert len(list((tmp_path/'R3').glob('*.npz')))==2
    with pytest.raises(FileExistsError):
        save_result(tmp_path,'R3','b0.5_complex',[])
