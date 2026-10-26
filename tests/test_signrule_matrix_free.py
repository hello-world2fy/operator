import numpy as np
from scipy.linalg import expm
from operator_walshness.signrule.models import tj_hamiltonian
from operator_walshness.signrule.geometry import chain
from operator_walshness.signrule.diagnostics import phase_entries
from operator_walshness.signrule.matrix_free import verified_ground,propagator_columns,sampled_column_entries,stochastic_thermal_diagonal


def test_sparse_propagator_and_partner_split_match_dense():
    h,*_=tj_hamiltonian(chain(4,True),t=2.)
    energies,vectors,checks=verified_ground(h)
    exact=expm(-.7*(h.toarray()-energies[0]*np.eye(h.shape[0])))
    columns=np.arange(h.shape[0]);actual=propagator_columns(h,.7,columns,energies[0])
    np.testing.assert_allclose(actual,exact,atol=2e-13)
    a=phase_entries(exact);b=sampled_column_entries(actual,columns)
    for key in ['row','col','phase','weight','test']:np.testing.assert_allclose(a[key],b[key],atol=1e-12)
    subset=sampled_column_entries(actual[:,[0,2]],[0,2])
    assert len(np.unique(subset['row']*len(h.toarray())+subset['col']))==len(subset['row'])


def test_thermal_diagonal_sampling_identity_is_exact():
    from scipy import sparse
    h=sparse.diags(np.arange(8,dtype=float),format='csr')
    diagonal,check,_=stochastic_thermal_diagonal(h,.8,0.,probes=32)
    expected=np.exp(-.8*np.arange(8));expected/=expected.sum()
    np.testing.assert_allclose(diagonal,expected,atol=1e-14)
    assert check['trace_standard_error']<1e-14
