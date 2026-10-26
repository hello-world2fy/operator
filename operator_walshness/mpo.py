"""Imaginary-time MPO for fixed-basis Operator Walshness.

The operator is kept as an MPO with tensors (χ_left, χ_right, ket, bra).
Gates act on the ket indices only, so the evolved operator is a Trotter
approximation to e^{-βH} rather than a purified density matrix. On-site fields
are split across the two bonds that touch a site, and edge sites keep the full
field, so each Pauli appears once.

The fixed-basis generating function is the contraction

    G(z) = Tr[O† B(z) O B(z)] / Tr[O† O],

with B(z) = ⊗ (|+><+| + z |-><-|). Polynomial coefficients in z are not formed.
G is evaluated either at z = e^α or on the roots of unity.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .hamiltonians import add_x, add_xx, add_yy, add_z, add_zz, check_n
from .walsh import spectrum_moments, tail_degree, walshness_from_spectrum


def identity_mpo(n: int) -> list[np.ndarray]:
    n = check_n(n)
    eye = np.eye(2, dtype=np.complex128).reshape(1, 1, 2, 2)
    return [eye.copy() for _ in range(n)]


def _field_fractions(n: int, bond: int) -> tuple[float, float]:
    left = 1.0 if bond == 0 else 0.5
    right = 1.0 if bond == n - 2 else 0.5
    return left, right


def bond_hamiltonian(
    model: str,
    n: int,
    bond: int,
    g: float = 1.0,
    h: float = 0.0,
    delta: float = 1.0,
) -> np.ndarray:
    """4×4 bond operator in the order ket = site + 2·(site+1)."""
    matrix = np.zeros((4, 4), dtype=np.float64)
    model = model.lower()
    if model in {"transverse_field_ising", "tfim"}:
        add_zz(matrix, 0, 1, -1.0)
        left, right = _field_fractions(n, bond)
        add_x(matrix, 0, -float(g) * left)
        add_x(matrix, 1, -float(g) * right)
        return matrix
    if model in {"mixed_field_ising", "mixed"}:
        add_zz(matrix, 0, 1, -1.0)
        left, right = _field_fractions(n, bond)
        add_x(matrix, 0, -float(g) * left)
        add_x(matrix, 1, -float(g) * right)
        add_z(matrix, 0, -float(h) * left)
        add_z(matrix, 1, -float(h) * right)
        return matrix
    if model in {"commuting_zz_ising", "zz", "classical_ising"}:
        add_zz(matrix, 0, 1, -1.0)
        return matrix
    if model in {"xxz", "xxz_symmetry_negative_control"}:
        add_xx(matrix, 0, 1, 1.0)
        add_yy(matrix, 0, 1, 1.0)
        add_zz(matrix, 0, 1, float(delta))
        return matrix
    if model in {"disconnected_dimers", "dimers"}:
        if bond % 2 == 0:
            add_xx(matrix, 0, 1, 0.25)
            add_yy(matrix, 0, 1, 0.25)
            add_zz(matrix, 0, 1, 0.25)
        return matrix
    raise ValueError(f"no MPO bond for {model}")


def onsite_hamiltonian(model: str, g: float = 1.0) -> np.ndarray | None:
    if model.lower() in {"one_site_product", "product_x", "independent_spins"}:
        matrix = np.zeros((2, 2), dtype=np.float64)
        add_x(matrix, 0, -float(g))
        return matrix
    return None


def apply_onsite_gate(tensors: list[np.ndarray], site: int, gate: np.ndarray) -> None:
    """Left-multiply a one-site gate on the ket index."""
    tensors[site] = np.einsum("Kk,lrkb->lrKb", np.asarray(gate, dtype=np.complex128), tensors[site], optimize=True)


def _gate_tensor(gate: np.ndarray) -> np.ndarray:
    """4×4 matrix with index ka+2 kb to a tensor [new_a, new_b, old_a, old_b]."""
    out = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    matrix = np.asarray(gate, dtype=np.complex128)
    for ka in range(2):
        for kb in range(2):
            for new_a in range(2):
                for new_b in range(2):
                    out[new_a, new_b, ka, kb] = matrix[new_a + 2 * new_b, ka + 2 * kb]
    return out


def _peak_normalize(tensor: np.ndarray) -> np.ndarray:
    """Drop a positive scale. Walsh ratios and discarded-weight ratios are invariant."""
    peak = float(np.max(np.abs(tensor)))
    if peak == 0.0 or not np.isfinite(peak):
        return tensor
    return tensor / peak


def _svd_factors(matrix: np.ndarray):
    """Full SVD. The divide-and-conquer driver can fail on a nearly low-rank bond."""
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd

        return svd(matrix, full_matrices=False, lapack_driver="gesvd", check_finite=False)


def _svd_values(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd

        _u, singular, _vh = svd(matrix, full_matrices=False, lapack_driver="gesvd", check_finite=False)
        return singular


def apply_bond_gate(
    tensors: list[np.ndarray],
    bond: int,
    gate: np.ndarray,
    chi_max: int,
    cutoff: float,
) -> float:
    """Left-multiply a two-site ket gate and compress the crossed bond.

    Returns the discarded Frobenius weight of this SVD, relative to the bond.
    """
    left = _peak_normalize(tensors[bond])
    right = _peak_normalize(tensors[bond + 1])
    joined = np.einsum("lmki,mrjn->lkijrn", left, right, optimize=True)
    updated = np.einsum("KJkj,lkijrn->lKiJnr", _gate_tensor(gate), joined, optimize=True)
    left_chi = left.shape[0]
    right_chi = right.shape[1]
    matrix = _peak_normalize(updated.reshape(left_chi * 4, 4 * right_chi))
    u_matrix, singular, vh_matrix = _svd_factors(matrix)
    weights = singular**2
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("SVD of a zero bond")
    keep = int(np.sum(weights / total > cutoff))
    keep = max(1, min(keep, chi_max, singular.size))
    discarded = float(weights[keep:].sum() / total) if keep < singular.size else 0.0
    kept_u = u_matrix[:, :keep]
    kept_s = singular[:keep]
    kept_vh = vh_matrix[:keep, :]
    tensors[bond] = kept_u.reshape(left_chi, 2, 2, keep).transpose(0, 3, 1, 2)
    scaled = kept_s[:, None] * kept_vh
    tensors[bond + 1] = scaled.reshape(keep, 2, 2, right_chi).transpose(0, 3, 1, 2)
    return discarded


def evolve_mpo(
    model: str,
    n: int,
    beta: float,
    dt: float,
    g: float = 1.0,
    h: float = 0.0,
    delta: float = 1.0,
    chi_max: int = 128,
    cutoff: float = 1e-12,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Second-order Trotter evolution of the identity up to e^{-βH}."""
    n = check_n(n)
    beta = float(beta)
    tensors = identity_mpo(n)
    stats = {"discarded_weight": 0.0, "largest_chi": 1, "dt": 0.0, "steps": 0}
    if beta == 0.0:
        return tensors, stats
    steps = max(1, int(round(beta / float(dt))))
    step = beta / steps
    stats["dt"] = step
    stats["steps"] = steps
    local = onsite_hamiltonian(model, g)
    if local is not None:
        gate = expm(-beta * local)
        for site in range(n):
            apply_onsite_gate(tensors, site, gate)
        return tensors, stats
    even = list(range(0, n - 1, 2))
    odd = list(range(1, n - 1, 2))
    half = {
        bond: expm(-0.5 * step * bond_hamiltonian(model, n, bond, g, h, delta)) for bond in range(n - 1)
    }
    full = {
        bond: expm(-step * bond_hamiltonian(model, n, bond, g, h, delta)) for bond in range(n - 1)
    }
    discarded = 0.0
    max_discarded = 0.0
    for _ in range(steps):
        for bond in even:
            weight = apply_bond_gate(tensors, bond, half[bond], chi_max, cutoff)
            discarded += weight
            max_discarded = max(max_discarded, weight)
        for bond in odd:
            weight = apply_bond_gate(tensors, bond, full[bond], chi_max, cutoff)
            discarded += weight
            max_discarded = max(max_discarded, weight)
        for bond in even:
            weight = apply_bond_gate(tensors, bond, half[bond], chi_max, cutoff)
            discarded += weight
            max_discarded = max(max_discarded, weight)
        _rescale_mpo(tensors)
    stats["discarded_weight"] = discarded
    stats["max_discarded_weight"] = max_discarded
    stats["largest_chi"] = max(tensor.shape[0] for tensor in tensors)
    stats["chi_profile"] = [int(tensor.shape[0]) for tensor in tensors[1:]]
    return tensors, stats


def _rescale_mpo(tensors: list[np.ndarray]) -> None:
    """Keep the first tensor O(1). Walsh ratios cancel this scale."""
    scale = float(np.max(np.abs(tensors[0])))
    if scale > 0.0 and np.isfinite(scale):
        tensors[0] = tensors[0] / scale


def to_dense(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract an MPO into a matrix with site 0 as the least significant bit."""
    n = len(tensors)
    if n > 12:
        raise ValueError("dense contraction is limited to n<=12")
    if tensors[0].shape[0] != 1:
        raise RuntimeError("left edge bond is not closed")
    acc = tensors[0][0]  # chi_right, ket, bra
    ket_dim = 2
    bra_dim = 2
    for tensor in tensors[1:]:
        grown = np.einsum("ckb,cRxy->Rkxby", acc, tensor, optimize=True)
        grown = grown.transpose(0, 2, 1, 4, 3)
        acc = grown.reshape(tensor.shape[1], ket_dim * 2, bra_dim * 2)
        ket_dim *= 2
        bra_dim *= 2
    if acc.shape[0] != 1:
        raise RuntimeError("MPO right bond did not close")
    return acc[0]


def _local_b(z: complex) -> np.ndarray:
    plus = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    minus = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)
    return plus + complex(z) * minus


def contraction(tensors: list[np.ndarray], z: complex) -> complex:
    """Tr[O† B(z) O B(z)], rescaled at each site."""
    value, log_scale = _scaled_contraction(tensors, z)
    return complex(value * np.exp(log_scale))


def _scaled_contraction(tensors: list[np.ndarray], z: complex) -> tuple[complex, float]:
    gate = _local_b(z)
    env = np.ones((1, 1), dtype=np.complex128)
    log_scale = 0.0
    for tensor in tensors:
        dagger = np.conjugate(tensor).transpose(0, 1, 3, 2)
        env = np.einsum("ab,aicd,de,bfeg,gc->if", env, dagger, gate, tensor, gate, optimize=True)
        scale = float(np.max(np.abs(env)))
        if scale == 0.0 or not np.isfinite(scale):
            return 0.0 + 0.0j, -np.inf
        env = env / scale
        log_scale += np.log(scale)
    return complex(env[0, 0]), log_scale


def log_generating_value(tensors: list[np.ndarray], z: complex) -> complex:
    numerator, log_num = _scaled_contraction(tensors, z)
    denominator, log_den = _scaled_contraction(tensors, 1.0 + 0.0j)
    return complex(np.log(numerator) + log_num - np.log(denominator) - log_den)


def fixed_walshness(tensors: list[np.ndarray], alpha: float) -> float:
    alpha = float(alpha)
    if alpha == 0.0:
        step = 1e-5
        plus = log_generating_value(tensors, np.exp(step)).real
        minus = log_generating_value(tensors, np.exp(-step)).real
        return float((plus - minus) / (2.0 * step))
    return float(log_generating_value(tensors, np.exp(alpha)).real / alpha)


def degree_spectrum(tensors: list[np.ndarray]) -> np.ndarray:
    n = len(tensors)
    length = 2 * n + 1
    angles = -2.0 * np.pi * np.arange(length) / length
    _, log_den = _scaled_contraction(tensors, 1.0 + 0.0j)
    samples = np.empty(length, dtype=np.complex128)
    for index, angle in enumerate(angles):
        value, log_num = _scaled_contraction(tensors, np.exp(1j * angle))
        samples[index] = value * np.exp(log_num - log_den)
    spectrum = np.fft.ifft(samples).real
    negative = float(np.sum(spectrum[spectrum < 0.0]))
    if negative < -1e-6:
        raise RuntimeError(f"MPO spectrum has negative mass {negative}")
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(spectrum.sum())
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"MPO spectrum normalized to {total}")
    return spectrum / total


def _as_mps(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Vectorize each site as s + 2 t, the grouping used for OSEE."""
    states = []
    for tensor in tensors:
        chi_left, chi_right, _, _ = tensor.shape
        physical = np.zeros((chi_left, chi_right, 4), dtype=np.complex128)
        for ket in range(2):
            for bra in range(2):
                physical[:, :, ket + 2 * bra] = tensor[:, :, ket, bra]
        states.append(physical)
    return states


def _left_canonicalize(states: list[np.ndarray], up_to: int) -> None:
    for site in range(up_to):
        left, right, physical = states[site].shape
        matrix = np.transpose(states[site], (0, 2, 1)).reshape(left * physical, right)
        orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
        states[site] = np.transpose(orthogonal.reshape(left, physical, orthogonal.shape[1]), (0, 2, 1))
        states[site + 1] = np.einsum("ab,bcd->acd", triangular, states[site + 1], optimize=True)


def _right_canonicalize(states: list[np.ndarray], down_to: int) -> None:
    for site in range(len(states) - 1, down_to, -1):
        left, right, physical = states[site].shape
        matrix = states[site].reshape(left, right * physical)
        orthogonal, triangular = np.linalg.qr(matrix.T, mode="reduced")
        states[site] = orthogonal.T.reshape(orthogonal.shape[1], right, physical)
        states[site - 1] = np.einsum("abc,bd->adc", states[site - 1], triangular.T, optimize=True)


def operator_entropy(tensors: list[np.ndarray], n_left: int | None = None) -> tuple[np.ndarray, float]:
    """Half-chain OSEE in bits from the canonical form of the vectorized MPO."""
    n = len(tensors)
    if n_left is None:
        n_left = n // 2
    if not 0 < n_left < n:
        raise ValueError("cut must separate the chain")
    states = _as_mps(tensors)
    # Sites n_left..n-1 become right-isometric, and sites 0..n_left-2 left-isometric,
    # so the singular values of site n_left-1 are the Schmidt spectrum.
    _right_canonicalize(states, n_left - 1)
    if n_left > 1:
        _left_canonicalize(states, n_left - 1)
    left, right, physical = states[n_left - 1].shape
    matrix = np.transpose(states[n_left - 1], (0, 2, 1)).reshape(left * physical, right)
    singular = _svd_values(matrix)
    weights = singular**2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("operator is zero")
    weights = weights / total
    positive = weights > 0.0
    entropy = float(-(weights[positive] * np.log2(weights[positive])).sum())
    return weights, entropy


def _rescale_env(env: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(env)))
    if scale == 0.0 or not np.isfinite(scale):
        return env, -np.inf
    return env / scale, float(np.log(scale))


def _site_trace(tensor: np.ndarray, operator: np.ndarray | None = None) -> np.ndarray:
    if operator is None:
        return np.trace(tensor, axis1=2, axis2=3)
    return np.einsum("bk,lrkb->lr", np.asarray(operator, dtype=np.complex128), tensor, optimize=True)


def _bond_trace(left: np.ndarray, right: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Trace a two-site ket operator. Flat index is site + 2·(site+1)."""
    pair = np.einsum("lmkb,mrjn->lrkbnj", left, right, optimize=True)
    weights = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    for ket0 in range(2):
        for ket1 in range(2):
            for bra0 in range(2):
                for bra1 in range(2):
                    weights[ket0, ket1, bra0, bra1] = operator[ket0 + 2 * ket1, bra0 + 2 * bra1]
    return np.einsum("lrkbnj,knbj->lr", pair, weights, optimize=True)


def thermal_observables(
    tensors: list[np.ndarray],
    model: str,
    g: float = 1.0,
    h: float = 0.0,
    delta: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Energy, purity and magnetizations of O/Tr(O), with O the stored MPO."""
    n = len(tensors)
    left_env: list[np.ndarray] = [np.ones((1, 1), dtype=np.complex128)]
    left_log = [0.0]
    for tensor in tensors:
        env, extra = _rescale_env(left_env[-1] @ _site_trace(tensor))
        left_env.append(env)
        left_log.append(left_log[-1] + extra)
    right_env: list[np.ndarray | None] = [None] * (n + 1)
    right_log = [0.0] * (n + 1)
    right_env[n] = np.ones((1, 1), dtype=np.complex128)
    for site in range(n - 1, -1, -1):
        env, extra = _rescale_env(_site_trace(tensors[site]) @ right_env[site + 1])
        right_env[site] = env
        right_log[site] = right_log[site + 1] + extra
    denominator = complex(left_env[n][0, 0]) * np.exp(left_log[n])

    def expectation(numerator: np.ndarray, log_scale: float) -> float:
        value = complex(numerator[0, 0]) * np.exp(log_scale) / denominator
        return float(value.real)

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    magnetization = np.zeros((n, 3), dtype=np.float64)
    for site in range(n):
        for axis, pauli in enumerate((pauli_x, pauli_y, pauli_z)):
            inserted = _site_trace(tensors[site], pauli)
            magnetization[site, axis] = expectation(
                left_env[site] @ inserted @ right_env[site + 1],
                left_log[site] + right_log[site + 1],
            )
    energy = 0.0
    local = onsite_hamiltonian(model, g)
    if local is not None:
        for site in range(n):
            inserted = _site_trace(tensors[site], local)
            energy += expectation(
                left_env[site] @ inserted @ right_env[site + 1],
                left_log[site] + right_log[site + 1],
            )
    else:
        for bond in range(n - 1):
            inserted = _bond_trace(
                tensors[bond],
                tensors[bond + 1],
                bond_hamiltonian(model, n, bond, g, h, delta),
            )
            energy += expectation(
                left_env[bond] @ inserted @ right_env[bond + 2],
                left_log[bond] + right_log[bond + 2],
            )
    frobenius_value, log_frobenius = _scaled_contraction(tensors, 1.0 + 0.0j)
    log_trace = float(np.log(complex(left_env[n][0, 0])).real + left_log[n])
    log_purity = float(np.log(complex(frobenius_value)).real + log_frobenius - 2.0 * log_trace)
    return {
        "energy": float(energy),
        "log_purity": log_purity,
        "magnetization": magnetization,
    }


def spectrum_diagnostics(spectrum: np.ndarray, alphas: tuple[float, ...], epsilons: tuple[float, ...]) -> dict:
    return {
        "walsh": np.array([walshness_from_spectrum(spectrum, alpha) for alpha in alphas]),
        "tails": np.array([tail_degree(spectrum, eps) for eps in epsilons]),
        "moments": spectrum_moments(spectrum),
    }
