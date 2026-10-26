import numpy as np
from operator_walshness.signrule.null_controls import involution_basis,same_spectrum_null


def test_random_conjugation_preserves_spectrum_and_selected_symmetry():
    rng=np.random.default_rng(12);p=np.arange(8)^7
    a=rng.normal(size=(8,8));rho=a@a.T;rho=(rho+rho[np.ix_(p,p)])/2;rho/=np.trace(rho)
    b,k=involution_basis(p)
    np.testing.assert_allclose(b.T@b,np.eye(8),atol=1e-14)
    assert k==4
    for family in ['complex_Haar','real_orthogonal_Haar','sector_Haar']:
        x,checks=same_spectrum_null(rho,family,311,p)
        np.testing.assert_allclose(np.linalg.eigvalsh(x),np.linalg.eigvalsh(rho),atol=1e-13)
        if family=='sector_Haar':assert checks['involution_commutator_max']<1e-13
        if family=='real_orthogonal_Haar':assert np.isrealobj(x)
