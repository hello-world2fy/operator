"""Identity checks for the Operator Walshness convention."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.hamiltonians import (
    Y,
    Z,
    build_hamiltonian,
    pauli_embedding,
    xxz_sector_hamiltonian,
)
from operator_walshness.observables import (
    operator_schmidt,
    thermal_observables,
    walsh_degree_correlations,
)
from operator_walshness.optimize import batched_moment, optimize_walshness
from operator_walshness.walsh import (
    align_direction_to_z,
    fixed_basis_spectrum,
    fixed_x_directions,
    fwht,
    gibbs_density,
    identity_spectrum,
    identity_walshness,
    intrinsic_alpha0,
    moment,
    moment_by_rotation,
    product_state_flat_spectrum,
    rotate_operator,
    state_walshness,
    walshness,
    walshness_from_directions,
    walshness_from_spectrum,
    x_basis_matrix,
)


def _embed(n: int, local: np.ndarray, site: int) -> np.ndarray:
    out = None
    eye = np.eye(2, dtype=local.dtype)
    for index in range(n - 1, -1, -1):
        factor = local if index == site else eye
        out = factor.copy() if out is None else np.kron(out, factor)
    return out


def _explicit_hadamard(n: int) -> np.ndarray:
    size = 1 << n
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    parity = np.bitwise_count(np.bitwise_and(rows, cols)).astype(np.int64) % 2
    return (1 - 2 * parity).astype(np.float64)


def test_pauli_embeddings_match_kronecker_products():
    n = 4
    for site in range(n):
        x = pauli_embedding(n, "X", site)
        z = pauli_embedding(n, "Z", site)
        y = pauli_embedding(n, "Y", site)
        assert np.allclose(x, _embed(n, np.array([[0, 1], [1, 0]], dtype=np.complex128), site))
        assert np.allclose(z, _embed(n, Z.astype(np.complex128), site))
        assert np.allclose(y, _embed(n, Y, site))


def test_yy_cancels_the_same_bit_flip_and_doubles_the_exchange():
    n = 2
    have = build_hamiltonian("xxz", n, delta=0.0)
    manual = _embed(n, np.array([[0, 1], [1, 0]], dtype=np.float64), 0)
    # Build XX+YY from Kronecker products in the same bit order.
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    xx = np.kron(x, x)
    yy = np.kron(y, y).real
    # kron(site1, site0): np.kron(A,B) has B fastest, and site 0 is fastest.
    # np.kron(x, x) is site1 ⊗ site0. Yes.
    assert np.allclose(xx + yy, have)


def test_dimer_spectrum_and_xxz_sector_agree_with_dense_diagonalization():
    dimer = build_hamiltonian("disconnected_dimers", 2)
    energies = np.linalg.eigvalsh(dimer)
    assert energies == pytest.approx([-0.75, 0.25, 0.25, 0.25])
    full = build_hamiltonian("xxz", 6, delta=0.5)
    pieces = []
    for down in range(7):
        block, _ = xxz_sector_hamiltonian(6, down, delta=0.5)
        pieces.append(np.linalg.eigvalsh(block))
    assert np.allclose(np.sort(np.concatenate(pieces)), np.linalg.eigvalsh(full))


def test_fwht_matches_the_character_matrix():
    hadamard = _explicit_hadamard(3)
    vector = np.random.default_rng(2).normal(size=8)
    assert np.allclose(fwht(vector), hadamard @ vector)
    matrix = np.random.default_rng(3).normal(size=(8, 8))
    unitary = hadamard / np.sqrt(8)
    assert np.allclose(x_basis_matrix(matrix), unitary @ matrix @ unitary)


def test_two_walsh_formulas_agree_for_nonhermitian_operators():
    rng = np.random.default_rng(4)
    operator = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    spectrum, diagnostics = fixed_basis_spectrum(operator)
    assert diagnostics["mass_relative_error"] < 1e-12
    assert diagnostics["probability_defect"] < 1e-12
    assert spectrum.min() >= -1e-15
    assert spectrum.sum() == pytest.approx(1.0)
    directions = fixed_x_directions(4)
    for alpha in (0.25, 0.5, 1.0, 2.0):
        from_spectrum = walshness_from_spectrum(spectrum, alpha)
        from_product = walshness_from_directions(operator, directions, alpha)
        from_rotation = np.log(moment_by_rotation(operator, directions, alpha)) / alpha
        assert from_product == pytest.approx(from_spectrum, rel=1e-9, abs=1e-10)
        assert from_rotation == pytest.approx(from_spectrum, rel=1e-9, abs=1e-10)
        value = moment(operator, directions, alpha)
        assert value == pytest.approx(np.exp(alpha * from_spectrum), rel=1e-9)


def test_analytic_identity_and_computational_projector():
    n = 5
    identity = np.eye(1 << n)
    spectrum, _ = fixed_basis_spectrum(identity)
    assert np.allclose(spectrum, identity_spectrum(n), atol=1e-12)
    for alpha in (0.0, 0.5, 1.0, 2.0):
        assert walshness(identity, alpha) == pytest.approx(identity_walshness(n, alpha), rel=1e-12, abs=1e-12)
        assert walshness(3.5 * identity, alpha) == pytest.approx(walshness(identity, alpha))
    projector = np.zeros((1 << n, 1 << n))
    projector[0, 0] = 1.0
    flat, _ = fixed_basis_spectrum(projector)
    assert np.allclose(flat, product_state_flat_spectrum(n), atol=1e-12)
    intrinsic, axes = intrinsic_alpha0(projector)
    assert intrinsic == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(axes[:, 2], 1.0)


def test_pure_state_identity_is_twice_the_state_walshness():
    rng = np.random.default_rng(5)
    for n in (1, 3, 4):
        state = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        state /= np.linalg.norm(state)
        projector = np.outer(state, np.conjugate(state))
        for alpha in (0.0, 0.3, 1.0, 2.0):
            assert walshness(projector, alpha) == pytest.approx(2.0 * state_walshness(state, alpha), rel=1e-9, abs=1e-10)


def test_alpha0_matches_magnetization_of_the_squared_state():
    hamiltonian = build_hamiltonian("mixed_field_ising", 5, g=1.05, h=0.5)
    evals, evecs = np.linalg.eigh(hamiltonian)
    density, probabilities = gibbs_density(evals, evecs, beta=0.7)
    spectrum, _ = fixed_basis_spectrum(density)
    fixed = walshness_from_spectrum(spectrum, 0.0)
    assert fixed == pytest.approx(walshness_from_directions(density, fixed_x_directions(5), 0.0), abs=1e-10)
    intrinsic, _ = intrinsic_alpha0(density)
    # The same number from the normalized Gibbs state at inverse temperature 2β.
    density_2b, _ = gibbs_density(evals, evecs, beta=1.4)
    bloch = []
    from operator_walshness.observables import magnetization_profile

    profile = magnetization_profile(density_2b)
    assert intrinsic == pytest.approx(np.sum(1.0 - np.linalg.norm(profile, axis=1)), abs=1e-10)
    assert fixed >= intrinsic - 1e-10
    # Infinite temperature is the identity result, independent of the model.
    infinite, _ = gibbs_density(evals, evecs, beta=0.0)
    assert walshness(infinite, 1.0) == pytest.approx(identity_walshness(5, 1.0), rel=1e-12)
    assert thermal_observables(evals, probabilities, 0.7)["purity"] == pytest.approx(np.trace(density @ density).real)


def test_local_alignment_implements_the_physical_basis_change():
    direction = np.array([0.3, -0.4, 0.5])
    direction /= np.linalg.norm(direction)
    unitary = align_direction_to_z(direction)
    pauli = direction[0] * np.array([[0, 1], [1, 0]]) + direction[1] * Y + direction[2] * Z
    assert np.allclose(unitary.conj().T @ pauli @ unitary, Z)


def test_intrinsic_value_is_invariant_under_a_physical_local_rotation_at_alpha0():
    hamiltonian = build_hamiltonian("tfim", 4, g=0.8)
    evals, evecs = np.linalg.eigh(hamiltonian)
    density, _ = gibbs_density(evals, evecs, beta=1.0)
    baseline, _ = intrinsic_alpha0(density)
    unitaries = [
        align_direction_to_z(vec)
        for vec in np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.6, 0.8]])
    ]
    rotated = rotate_operator(density, unitaries)
    assert np.allclose(rotated, rotated.conj().T, atol=1e-10)
    transformed, _ = intrinsic_alpha0(rotated)
    assert transformed == pytest.approx(baseline, abs=1e-9)
    before, entropy_before = operator_schmidt(density)
    after, entropy_after = operator_schmidt(rotated)
    assert entropy_after == pytest.approx(entropy_before, abs=1e-9)
    assert np.allclose(np.sort(before), np.sort(after), atol=1e-8)


def test_operator_schmidt_grouping_matches_an_explicit_tensor():
    rng = np.random.default_rng(6)
    operator = rng.normal(size=(8, 8))
    operator = 0.5 * (operator + operator.T)
    weights, _ = operator_schmidt(operator, n_left=1)
    psi = np.zeros((4, 4, 4))
    for s in range(8):
        for t in range(8):
            local = []
            for site in range(3):
                local.append(((s >> site) & 1) + 2 * ((t >> site) & 1))
            # Axis 0 is the slow site (site 2), axis 2 is site 0.
            psi[local[2], local[1], local[0]] = operator[s, t]
    # Site 0 is the fast axis, matching a cut that keeps one site on the left.
    manual = np.linalg.svd(psi.reshape(16, 4), compute_uv=False)
    manual = manual**2
    manual /= manual.sum()
    assert np.allclose(np.sort(weights), np.sort(manual), atol=1e-10)


def test_degree_correlation_matches_the_explicit_distribution():
    rng = np.random.default_rng(7)
    generator = rng.normal(size=(4, 4))
    density = generator @ generator.T
    density /= np.trace(density)
    spectrum, _ = fixed_basis_spectrum(density)
    stats = walsh_degree_correlations(density)
    assert np.sum(stats["degree_mean"]) == pytest.approx(np.dot(np.arange(spectrum.size), spectrum), abs=1e-9)
    # Explicit joint degree bits for n=2.
    matrix = x_basis_matrix(density)
    weights = np.abs(matrix) ** 2
    weights /= weights.sum()
    mean = np.zeros(2)
    corr = np.zeros((2, 2))
    for r in range(4):
        for q in range(4):
            bits = [((r >> i) & 1) + ((q >> i) & 1) for i in range(2)]
            for i in range(2):
                mean[i] += bits[i] * weights[r, q]
                for j in range(2):
                    corr[i, j] += bits[i] * bits[j] * weights[r, q]
    connected = corr - np.outer(mean, mean)
    assert np.allclose(stats["degree_mean"], mean, atol=1e-9)
    assert np.allclose(stats["degree_correlation"], connected, atol=1e-9)


def test_torch_moment_matches_numpy_and_a_finite_difference():
    import torch

    rng = np.random.default_rng(8)
    operator = rng.normal(size=(8, 8))
    operator = 0.5 * (operator + operator.T)
    directions = rng.normal(size=(3, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    alpha = 0.8
    numpy_value = moment(operator, directions, alpha)
    op = torch.tensor(operator, dtype=torch.complex128)
    batched = batched_moment(op, torch.tensor(directions, dtype=torch.float64).unsqueeze(0), alpha)
    assert float(batched) == pytest.approx(numpy_value, rel=1e-10)
    # One-site finite difference along a tangent.
    step = 1e-6
    base = directions.copy()
    moved = base.copy()
    tangent = np.array([0.0, 1.0, 0.0])
    tangent -= np.dot(tangent, base[1]) * base[1]
    tangent /= np.linalg.norm(tangent)
    moved[1] = base[1] * np.cos(step) + tangent * np.sin(step)
    numerical = (moment(operator, moved, alpha) - numpy_value) / step
    plus = base.copy()
    plus[1] = base[1] * np.cos(step) + tangent * np.sin(step)
    minus = base.copy()
    minus[1] = base[1] * np.cos(step) - tangent * np.sin(step)
    numerical = (moment(operator, plus, alpha) - moment(operator, minus, alpha)) / (2 * step)
    direction = torch.tensor(base, dtype=torch.float64, requires_grad=True)
    value = batched_moment(op, direction.unsqueeze(0), alpha)
    value.backward()
    gradient = direction.grad.numpy()
    analytic = np.dot(gradient[1], tangent)
    assert analytic == pytest.approx(numerical, rel=1e-5, abs=1e-7)


def test_optimizer_recovers_a_product_ground_projector_and_respects_the_fixed_basis():
    projector = np.zeros((4, 4))
    projector[0, 0] = 1.0
    found = optimize_walshness(projector, alpha=1.0, restarts=8, seed=0, steps=80, polish=15, device="cpu")
    assert found["W_upper"] == pytest.approx(0.0, abs=1e-7)
    assert found["W_upper"] <= found["fixed_W"] + 1e-10
    assert found["moment_disagreement"] < 1e-8
    assert found["certified_global"] is False
    zero = optimize_walshness(projector, alpha=0.0)
    assert zero["certified_global"] is True
    assert zero["W_upper"] == pytest.approx(0.0, abs=1e-12)


def test_product_additivity_of_independent_dimers():
    one = build_hamiltonian("disconnected_dimers", 2)
    two = build_hamiltonian("disconnected_dimers", 4)
    evals_one, vecs_one = np.linalg.eigh(one)
    evals_two, vecs_two = np.linalg.eigh(two)
    rho_one, _ = gibbs_density(evals_one, vecs_one, 1.25)
    rho_two, _ = gibbs_density(evals_two, vecs_two, 1.25)
    for alpha in (0.0, 1.0, 2.0):
        assert walshness(rho_two, alpha) == pytest.approx(2.0 * walshness(rho_one, alpha), rel=1e-9, abs=1e-9)
