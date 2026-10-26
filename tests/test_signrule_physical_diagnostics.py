import numpy as np
import pytest
from scipy.linalg import expm
from operator_walshness.signrule.geometry import chain,square
from operator_walshness.signrule.models import heisenberg,tj_hamiltonian,block_eigensystem,block_gibbs
from operator_walshness.signrule.physical_diagnostics import (worldline_phase_severity,tj_binary_encoding,
    constrained_linear_fit,constrained_operator_entropy,qutrit_full_spectrum,spin_hole_correlations,sparse_operator_entropy)
from operator_walshness.signrule.diagnostics import phase_entries,score_phase
from operator_walshness.signrule.blind import qutrit_moment
from operator_walshness.walsh import walshness_from_spectrum


def test_worldline_phase_ratio_gauge_invariance_and_complex_extension():
    h=heisenberg(chain(4),.5).toarray();phase=np.exp(1j*np.array([.17*i*i for i in range(16)]))
    transformed=phase[:,None]*h*phase[None,:].conj()
    reference=worldline_phase_severity(h,.8,4);actual=worldline_phase_severity(transformed,.8,4)
    assert reference['gamma']>=0
    assert actual['gamma']==pytest.approx(reference['gamma'],abs=1e-12)
    assert actual['complex_basis']
    assert worldline_phase_severity(heisenberg(chain(4)).toarray(),1,4)['gamma']==pytest.approx(0,abs=1e-12)


def test_controlled_frustration_has_an_explicit_local_stoquastic_basis():
    from operator_walshness.signrule.models import frustrated
    from operator_walshness.walsh import rotate_operator
    hadamard=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    for lam in [0.,.05,.1,.2,.4]:
        h=rotate_operator(frustrated(chain(4),lam).toarray(),[hadamard]*4)
        off=h.copy();np.fill_diagonal(off,0)
        assert np.max(off.real)<1e-13 and np.max(abs(off.imag))<1e-13
        assert worldline_phase_severity(h,2,4)['gamma']==pytest.approx(0,abs=1e-12)


def test_constrained_linear_gauge_recovered_without_qubit_W_interpretation():
    _,digits,*_=tj_hamiltonian(chain(4));codes=tj_binary_encoding(digits)
    phase=1.-2*(np.bitwise_count(codes.astype(np.uint64)&0b10010101)%2).astype(float)
    matrix=phase[:,None]*phase[None,:]
    entries=phase_entries(matrix);f,_=constrained_linear_fit(entries,codes)
    assert score_phase(entries,f)['circular_loss']==pytest.approx(0,abs=1e-12)


def test_qutrit_exact_spectrum_matches_kernel_moments_and_OSEE_embedding():
    h,digits,*_=tj_hamiltonian(chain(4));rho,_=block_gibbs(block_eigensystem(h),.6,len(digits))
    p=qutrit_full_spectrum(rho,digits)
    for alpha in [.5,1,2]:
        assert walshness_from_spectrum(p,alpha)==pytest.approx(np.log(qutrit_moment(rho,digits,alpha))/alpha,abs=1e-10)
    result=constrained_operator_entropy(rho,digits)
    codes=digits@(3**np.arange(4));full=np.zeros((81,81));full[np.ix_(codes,codes)]=rho
    reshaped=full.reshape(9,9,9,9).transpose(0,2,1,3).reshape(81,81)
    weights=np.linalg.svd(reshaped,compute_uv=False)**2/np.vdot(rho,rho).real
    expected=-weights[weights>0]@np.log2(weights[weights>0])
    assert result['OSEE']==pytest.approx(expected,abs=1e-11)
    corr=spin_hole_correlations(rho,digits)
    assert corr['hole_density'].sum()==pytest.approx(1)
    assert np.max(abs(np.diag(corr['hole_spin_raw'])))==0


def test_rotated_qutrit_spectrum_matches_constrained_complex_kernel():
    h,digits,*_=tj_hamiltonian(chain(4));rho,_=block_gibbs(block_eigensystem(h),.6,len(digits))
    rng=np.random.default_rng(82);unitaries=[]
    for _ in range(4):
        q=rng.normal(size=4);q/=np.linalg.norm(q);a=q[0]+1j*q[1];b=q[2]+1j*q[3]
        u=np.eye(3,dtype=complex);u[0,0]=np.exp(.37j);u[1:,1:]=[[a,b],[-b.conjugate(),a.conjugate()]];unitaries.append(u)
    spectrum=qutrit_full_spectrum(rho,digits,unitaries)
    for alpha in [.5,1,2]:
        assert walshness_from_spectrum(spectrum,alpha)==pytest.approx(np.log(qutrit_moment(rho,digits,alpha,unitaries))/alpha,abs=1e-11)


def test_batched_qutrit_kernel_keeps_exponential_identity_in_float64():
    import torch
    from operator_walshness.signrule.blind import qutrit_batched_moment,qutrit_local_vectors
    h,digits,*_=tj_hamiltonian(chain(4));rho,_=block_gibbs(block_eigensystem(h),.6,len(digits))
    rng=np.random.default_rng(65);raw=rng.normal(size=(1,4,4));unitaries=[]
    for q in raw[0]:
        q=q/np.linalg.norm(q);a=q[0]+1j*q[1];b=q[2]+1j*q[3]
        u=np.eye(3,dtype=complex);u[1:,1:]=[[a,b],[-b.conjugate(),a.conjugate()]];unitaries.append(u)
    for alpha in [.5,1,2]:
        value=qutrit_batched_moment(torch.tensor(rho,dtype=torch.complex128),torch.tensor(digits,dtype=torch.int64),alpha,
            qutrit_local_vectors(torch.tensor(raw,dtype=torch.float64)))
        assert float(value[0])==pytest.approx(qutrit_moment(rho,digits,alpha,unitaries),rel=1e-13)


def test_sparse_spatial_Gram_entropy_matches_dense_SVD():
    from scipy import sparse
    from operator_walshness.round2 import schmidt
    rng=np.random.default_rng(39);operator=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
    operator[abs(operator)<1.2]=0
    actual=sparse_operator_entropy(sparse.csr_matrix(operator),6)
    expected=schmidt(operator)[1]
    assert actual['OSEE']==pytest.approx(expected,abs=1e-11)
def test_qutrit_degree_derivatives_match_full_fourier_histogram():
    from operator_walshness.signrule.physical_diagnostics import qutrit_degree_moments,qutrit_full_spectrum
    from operator_walshness.signrule.models import tj_hamiltonian,block_eigensystem,block_gibbs
    from operator_walshness.signrule.geometry import chain
    import numpy as np
    h,digits,*_=tj_hamiltonian(chain(3),t=1.4)
    rho,_=block_gibbs(block_eigensystem(h),.8,len(digits))
    p=qutrit_full_spectrum(rho,digits);k=np.arange(len(p));m=qutrit_degree_moments(rho,digits)
    np.testing.assert_allclose(m['Z3_degree_mean'].sum(),p@k,atol=1e-12)
    np.testing.assert_allclose(m['Z3_degree_correlation'].sum(),p@(k*k)-(p@k)**2,atol=1e-11)
