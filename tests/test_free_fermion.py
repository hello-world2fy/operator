"""Free-fermion TFIM against exact diagonalization, including the low-temperature regime."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize, gibbs_record
from operator_walshness.free_fermion import (
    OpenBoundaryTFIM,
    degree_spectrum,
    fixed_walshness,
    local_magnetization,
    log_partition,
)
from operator_walshness.integrable import convolve_copies
from operator_walshness.walsh import identity_spectrum


def test_partition_spectrum_and_walshness_match_ed_down_to_low_temperature():
    model = OpenBoundaryTFIM(6, 1.2)
    evals, evecs, _ = diagonalize("transverse_field_ising", 6, "open", {"g": 1.2})
    for beta in (0.0, 0.5, 2.0, 8.0, 16.0):
        assert log_partition(6, 1.2, beta) == pytest.approx(
            np.log(np.sum(np.exp(-beta * (evals - evals.max())))) - beta * evals.max(), abs=1e-9
        )
        record = gibbs_record(evals, evecs, beta)
        for alpha, value in zip((0.0, 0.5, 1.0, 2.0), record["W_fixed"][[0, 2, 3, 4]]):
            assert model.fixed_walshness(beta, alpha) == pytest.approx(value, rel=1e-8, abs=1e-8)
        assert np.allclose(model.degree_spectrum(beta), record["spectrum"], atol=1e-8)
        assert model.intrinsic_alpha0(beta) == pytest.approx(record["W_alpha0_intrinsic"], abs=1e-8)


def test_local_magnetization_matches_the_gibbs_profile():
    evals, evecs, _ = diagonalize("transverse_field_ising", 5, "open", {"g": 0.8})
    record = gibbs_record(evals, evecs, 1.1)
    assert np.allclose(local_magnetization(5, 0.8, 1.1), record["magnetization"], atol=1e-8)
    assert fixed_walshness(8, 1.0, 16.0, 1.0) == pytest.approx(
        gibbs_record(*diagonalize("transverse_field_ising", 8, "open", {"g": 1.0})[:2], 16.0)["W_fixed"][3],
        rel=1e-8,
    )


def test_dimer_spectra_convolve():
    small_evals, small_evecs, _ = diagonalize("disconnected_dimers", 2, "open", {})
    large_evals, large_evecs, _ = diagonalize("disconnected_dimers", 4, "open", {})
    beta = 1.25
    one = gibbs_record(small_evals, small_evecs, beta)["spectrum"]
    two = gibbs_record(large_evals, large_evecs, beta)["spectrum"]
    assert np.allclose(convolve_copies(one, 2), two, atol=1e-10)
    assert np.allclose(degree_spectrum(4, 1.5, 0.0), identity_spectrum(4), atol=1e-10)
