"""Blocking regressions for round 2, including complex shared-basis rotations."""
import numpy as np
import pytest

from operator_walshness.hamiltonians import build_hamiltonian
from operator_walshness.optimize import optimize_walshness
from operator_walshness.observables import walsh_degree_correlations, degree_correlations_from_x_matrix
from operator_walshness.walsh import (
    directions_to_walsh_basis, fixed_basis_spectrum, gibbs_density,
    rotate_operator, walshness_from_directions, walshness_from_spectrum,
    x_basis_matrix, intrinsic_alpha0,
)


@pytest.mark.parametrize('n', [1, 3])
@pytest.mark.parametrize('kind', ['real', 'complex', 'hermitian', 'gibbs', 'projector'])
def test_moment_matches_rotated_fixed_spectrum(n, kind):
    rng = np.random.default_rng(1200+n)
    d = 2**n
    op = rng.normal(size=(d,d))
    if kind != 'real':
        op = op + 1j*rng.normal(size=(d,d))
    if kind in ('hermitian','gibbs'):
        op = (op+op.conj().T)/2
    if kind == 'gibbs':
        e,v = np.linalg.eigh(op)
        op,_ = gibbs_density(e,v,0.7)
        assert np.max(abs(op.imag)) > 1e-4
    if kind == 'projector':
        state = op[:,0]/np.linalg.norm(op[:,0])
        op = np.outer(state,state.conj())
    axes = rng.normal(size=(n,3))
    axes /= np.linalg.norm(axes,axis=1,keepdims=True)
    rotated = rotate_operator(op,directions_to_walsh_basis(axes))
    spectrum,diag = fixed_basis_spectrum(rotated)
    assert diag['min_bin'] >= -1e-12
    assert diag['probability_defect'] < 1e-10
    for alpha in (0.,0.25,0.5,1.,2.):
        assert abs(walshness_from_spectrum(spectrum,alpha)-walshness_from_directions(op,axes,alpha)) < 1e-8


def test_optimized_gibbs_uses_the_same_physical_basis():
    e,v = np.linalg.eigh(build_hamiltonian('mixed_field_ising',3,g=1.05,h=0.5))
    rho,_ = gibbs_density(e,v,0.5)
    best = optimize_walshness(rho,1.,restarts=8,steps=40,polish=20)
    rotated = rotate_operator(rho,directions_to_walsh_basis(best['axes_best']))
    p,_ = fixed_basis_spectrum(rotated)
    assert abs(walshness_from_spectrum(p,1)-best['W_upper']) < 1e-8
    value,axes = intrinsic_alpha0(rho)
    p,_ = fixed_basis_spectrum(rotate_operator(rho,directions_to_walsh_basis(axes)))
    assert abs(value-walshness_from_spectrum(p,0)) < 1e-10


def test_complex_spatial_formula_matches_probability_covariance():
    rng = np.random.default_rng(21)
    a = rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    rho = a@a.conj().T
    direct = degree_correlations_from_x_matrix(x_basis_matrix(rho))
    formula = walsh_degree_correlations(rho)
    np.testing.assert_allclose(direct['degree_correlation'],formula['degree_correlation'],atol=1e-10)
