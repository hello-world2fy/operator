"""Large-n exact routes against dense diagonalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.hamiltonians import build_hamiltonian
from operator_walshness.integrable import (
    classical_chain_spectrum,
    classical_chain_walshness,
    classical_diagonal_spectrum_from_boltzmann,
    classical_chain_boltzmann,
    classical_staggered_z_walshness,
    cylinder_spectrum,
    cylinder_walshness,
    ising_2d_boltzmann,
    product_x_spectrum,
)
from operator_walshness.walsh import (
    fixed_basis_spectrum,
    gibbs_density,
    identity_spectrum,
    identity_walshness,
    walshness_from_spectrum,
)


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_classical_chain_matches_dense_gibbs_and_the_identity(boundary):
    n = 6
    beta = 0.8
    weights = classical_chain_boltzmann(n, beta, boundary)
    weights /= weights.sum()
    from_diagonal = classical_diagonal_spectrum_from_boltzmann(weights)
    from_transfer = classical_chain_spectrum(n, beta, boundary)
    hamiltonian = build_hamiltonian("commuting_zz_ising", n, boundary=boundary)
    evals, evecs = np.linalg.eigh(hamiltonian)
    density, _ = gibbs_density(evals, evecs, beta)
    dense, diagnostics = fixed_basis_spectrum(density)
    assert diagnostics["mass_relative_error"] < 1e-10
    assert np.allclose(from_diagonal, dense, atol=1e-10)
    assert np.allclose(from_transfer, dense, atol=1e-8)
    infinite = classical_chain_spectrum(32, 0.0, boundary)
    assert np.allclose(infinite, identity_spectrum(32), atol=1e-10)
    from operator_walshness.integrable import classical_chain_walshness

    assert classical_chain_walshness(512, 0.0, 1.0, boundary) == pytest.approx(
        identity_walshness(512, 1.0), rel=1e-12
    )


def test_product_transverse_spins_match_dense_and_add():
    n = 5
    g = 0.7
    beta = 1.3
    hamiltonian = build_hamiltonian("one_site_product", n, g=g)
    evals, evecs = np.linalg.eigh(hamiltonian)
    density, _ = gibbs_density(evals, evecs, beta)
    dense, _ = fixed_basis_spectrum(density)
    assert np.allclose(product_x_spectrum(n, g, beta), dense, atol=1e-10)


def test_open_rectangle_and_cylinder_reduce_to_the_diagonal_formula():
    weights = ising_2d_boltzmann(3, 2, beta=0.4)
    from_diag = classical_diagonal_spectrum_from_boltzmann(weights)
    # 3x2 open rectangle is not a cylinder. Check the diagonal formula against ED.
    from operator_walshness.hamiltonians import ising_2d_hamiltonian

    hamiltonian = ising_2d_hamiltonian(3, 2)
    evals, evecs = np.linalg.eigh(hamiltonian)
    density, _ = gibbs_density(evals, evecs, 0.4)
    dense, _ = fixed_basis_spectrum(density)
    assert np.allclose(from_diag, dense, atol=1e-10)
    # Width-2 cylinder of length 2 is a 4-site plaquette with vertical PBC.
    cylinder = cylinder_spectrum(2, 2, beta=0.4)
    idx = np.arange(16)
    # site = x + 2*y, circumference along x, length along y. Each edge once.
    energy = np.zeros(16)
    spin = lambda s, i: 1 - 2 * ((s >> i) & 1)
    for y in range(2):
        energy += -spin(idx, 2 * y) * spin(idx, 1 + 2 * y)
    for x in range(2):
        energy += -spin(idx, x) * spin(idx, x + 2)
    explicit = classical_diagonal_spectrum_from_boltzmann(np.exp(-0.4 * energy))
    assert np.allclose(cylinder, explicit, atol=1e-8)
    assert cylinder_walshness(4, 2, 0.0, 1.0) == pytest.approx(identity_walshness(8, 1.0))
    assert cylinder_walshness(2, 2, 0.4, 1.0) == pytest.approx(
        walshness_from_spectrum(explicit, 1.0), abs=1e-8
    )
    cooled = cylinder_walshness(32, 4, 16.0, 1.0)
    assert np.isfinite(cooled)


def test_staggered_z_matches_identity_and_dense_moment():
    from operator_walshness.exact import diagonalize
    from operator_walshness.optimize import _staggered
    from operator_walshness.walsh import walshness_from_directions

    for boundary in ("open", "periodic"):
        assert classical_staggered_z_walshness(12, 0.0, 1.0, boundary) == pytest.approx(
            identity_walshness(12, 1.0)
        )
        assert classical_staggered_z_walshness(8, 16.0, 1.0, boundary) == pytest.approx(8.0, abs=1e-10)
    evals, evecs, _ = diagonalize("commuting_zz_ising", 6, "open", {})
    density, _ = gibbs_density(evals, evecs, 1.0)
    axes = _staggered(6, 2)
    dense = walshness_from_directions(density, axes, 1.0)
    assert classical_staggered_z_walshness(6, 1.0, 1.0, "open") == pytest.approx(dense, abs=1e-10)
    # Fixed X is a different basis. Both are above the floor n at this temperature.
    fixed = classical_chain_walshness(6, 1.0, 1.0, "open")
    staggered = classical_staggered_z_walshness(6, 1.0, 1.0, "open")
    assert staggered < fixed
    assert staggered > 6.0
