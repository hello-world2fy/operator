"""Hutchinson estimate against the exact fixed-basis value."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize, gibbs_record
from operator_walshness.hamiltonians import build_hamiltonian
from operator_walshness.stochastic import estimate_walshness
from operator_walshness.walsh import identity_walshness


def test_identity_estimate_agrees_within_the_sampling_error():
    hamiltonian = np.zeros((32, 32))
    estimate = estimate_walshness(hamiltonian, beta=1.0, alpha=1.0, probes=256, seed=0, batches=8)
    exact = identity_walshness(5, 1.0)
    assert estimate["W"] == pytest.approx(exact, abs=3.0 * estimate["W_stderr"] + 1e-3)


def test_tfim_estimate_tracks_exact_diagonalization():
    evals, evecs, _ = diagonalize("transverse_field_ising", 6, "open", {"g": 1.0})
    exact = gibbs_record(evals, evecs, 0.5)["W_fixed"][3]
    estimate = estimate_walshness(
        build_hamiltonian("transverse_field_ising", 6, g=1.0),
        beta=0.5,
        alpha=1.0,
        probes=128,
        seed=1,
        batches=4,
    )
    assert estimate["W"] == pytest.approx(exact, abs=4.0 * estimate["W_stderr"] + 1e-2)
    assert estimate["moment_stderr"] > 0.0
