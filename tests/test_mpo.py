"""MPO evolution and the fixed-basis contraction against exact diagonalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize, gibbs_record
from operator_walshness.mpo import (
    degree_spectrum,
    evolve_mpo,
    fixed_walshness,
    identity_mpo,
    operator_entropy,
    thermal_observables,
    to_dense,
)
from operator_walshness.observables import operator_schmidt
from operator_walshness.walsh import identity_walshness


def test_low_temperature_mpo_stays_finite():
    tensors, _stats = evolve_mpo("transverse_field_ising", 6, beta=16.0, dt=0.1, g=1.0, chi_max=32, cutoff=1e-12)
    value = fixed_walshness(tensors, 1.0)
    assert np.isfinite(value)
    spectrum = degree_spectrum(tensors)
    assert np.all(np.isfinite(spectrum))


def test_identity_contraction_is_the_analytic_walshness():
    tensors = identity_mpo(5)
    assert np.allclose(to_dense(tensors), np.eye(32))
    for alpha in (0.0, 0.5, 1.0, 2.0):
        assert fixed_walshness(tensors, alpha) == pytest.approx(identity_walshness(5, alpha), rel=1e-10, abs=1e-10)


def test_product_field_is_exact():
    tensors, stats = evolve_mpo("one_site_product", 6, beta=1.3, dt=0.2, g=0.7)
    assert stats["largest_chi"] == 1
    evals, evecs, _ = diagonalize("one_site_product", 6, "open", {"g": 0.7})
    record = gibbs_record(evals, evecs, 1.3)
    assert fixed_walshness(tensors, 1.0) == pytest.approx(record["W_fixed"][3], rel=1e-10)
    assert np.allclose(degree_spectrum(tensors), record["spectrum"], atol=1e-10)


def test_tfim_trotter_matches_exact_diagonalization_when_uncompressed():
    beta = 0.8
    coarse, _ = evolve_mpo("transverse_field_ising", 6, beta=beta, dt=0.05, g=1.0, chi_max=64, cutoff=1e-14)
    tensors, stats = evolve_mpo("transverse_field_ising", 6, beta=beta, dt=0.01, g=1.0, chi_max=64, cutoff=1e-14)
    assert stats["discarded_weight"] < 1e-8
    evals, evecs, _ = diagonalize("transverse_field_ising", 6, "open", {"g": 1.0})
    record = gibbs_record(evals, evecs, beta)
    assert np.allclose(degree_spectrum(tensors), record["spectrum"], atol=2e-5)
    assert fixed_walshness(tensors, 1.0) == pytest.approx(record["W_fixed"][3], rel=1e-4, abs=1e-4)
    # Second-order Trotter: halving the step cuts the Walsh error by more than a factor of two.
    coarse_error = abs(fixed_walshness(coarse, 1.0) - record["W_fixed"][3])
    fine_error = abs(fixed_walshness(tensors, 1.0) - record["W_fixed"][3])
    assert fine_error < coarse_error / 2.0
    weights, entropy = operator_entropy(tensors)
    dense_weights, dense_entropy = operator_schmidt(to_dense(tensors))
    assert entropy == pytest.approx(dense_entropy, abs=1e-8)
    count = min(weights.size, dense_weights.size)
    assert np.allclose(weights[:count], dense_weights[:count], atol=1e-8)
    thermal = thermal_observables(tensors, "transverse_field_ising", g=1.0)
    assert thermal["energy"] == pytest.approx(record["thermo"]["energy"], rel=1e-4, abs=1e-4)
    assert thermal["log_purity"] == pytest.approx(np.log(record["thermo"]["purity"]), rel=1e-4, abs=1e-4)
    assert np.allclose(thermal["magnetization"], record["magnetization"], atol=1e-4)


def test_low_temperature_evolution_stays_finite():
    tensors, _stats = evolve_mpo("mixed_field_ising", 8, beta=8.0, dt=0.1, g=1.05, h=0.5, chi_max=32)
    value = fixed_walshness(tensors, 1.0)
    assert np.isfinite(value)
