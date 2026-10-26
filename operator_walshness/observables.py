"""Thermal observables, Walsh-degree correlations, and operator entanglement."""

from __future__ import annotations

import numpy as np

from .walsh import frobenius_norm_squared, hamming_weights, trace_pauli


def thermal_observables(evals: np.ndarray, probabilities: np.ndarray, beta: float) -> dict[str, float]:
    """Energy, heat capacity, purity, and effective dimension from the spectrum."""
    energy = float(np.dot(probabilities, evals))
    second = float(np.dot(probabilities, evals**2))
    variance = second - energy**2
    purity = float(np.dot(probabilities, probabilities))
    return {
        "energy": energy,
        "energy_variance": variance,
        "heat_capacity": float(beta**2 * variance),
        "purity": purity,
        "r_eff": float(1.0 / purity) if purity > 0.0 else float("inf"),
    }


def distance_correlations(density: np.ndarray, pauli: str) -> np.ndarray:
    """Translationally averaged connected <P_i P_{i+d}> - <P_i><P_{i+d}> for d=0..n-1."""
    n = int(np.log2(density.shape[0]))
    one = np.array([(trace_pauli(density, pauli, site)).real for site in range(n)])
    # Density matrices used here are normalized, so the trace already is the expectation.
    corr = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    size = density.shape[0]
    idx = np.arange(size)
    diag = np.diag(density).real
    for i in range(n):
        for d in range(n - i):
            j = i + d
            if pauli == "Z":
                zi = 1 - 2 * ((idx >> i) & 1)
                zj = 1 - 2 * ((idx >> j) & 1)
                raw = float(np.dot(diag, zi * zj))
            elif pauli == "X":
                raw = float(np.sum(density[idx, idx ^ (1 << i) ^ (1 << j)]).real)
            else:
                raise ValueError("distance correlations support X and Z")
            corr[d] += raw - one[i] * one[j]
            counts[d] += 1.0
    return corr / counts


def ensemble_magnetization(eigenvectors: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Single-site Bloch vector of a mixture sum_k p_k |k><k|.

    Eigenvectors are real columns. Y expectations vanish for real states.
    """
    n = int(np.log2(eigenvectors.shape[0]))
    size = eigenvectors.shape[0]
    idx = np.arange(size)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    profile = np.zeros((n, 3), dtype=np.float64)
    diag = np.sum(probabilities * eigenvectors**2, axis=1)
    for site in range(n):
        bit = (idx >> site) & 1
        profile[site, 2] = float(np.dot(diag, 1 - 2 * bit))
        flipped = eigenvectors[idx ^ (1 << site), :]
        local = np.sum(eigenvectors * flipped, axis=0)
        profile[site, 0] = float(np.dot(probabilities, local))
    return profile


def degree_correlations_from_x_matrix(x_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Local degree means and connected correlations from H O H.

    Bits of the row index are r and bits of the column index are q.
    """
    n = int(np.log2(x_matrix.shape[0]))
    weights = np.abs(x_matrix) ** 2
    weights /= float(weights.sum())
    bits = np.stack(
        [((np.arange(weights.shape[0]) >> site) & 1).astype(np.float64) for site in range(n)],
        axis=0,
    )
    row_marginal = weights.sum(axis=1)
    col_marginal = weights.sum(axis=0)
    expect_r = bits @ row_marginal
    expect_q = bits @ col_marginal
    mean = expect_r + expect_q
    rr = (bits * row_marginal) @ bits.T
    qq = (bits * col_marginal) @ bits.T
    rq = bits @ weights @ bits.T
    raw = rr + rq + rq.T + qq
    return {"degree_mean": mean, "degree_correlation": raw - np.outer(mean, mean), "degree_second": raw}


def magnetization_profile(density: np.ndarray) -> np.ndarray:
    """Return shape (n, 3) expectations of X, Y, Z. ``density`` must have trace one."""
    n = int(np.log2(density.shape[0]))
    profile = np.zeros((n, 3), dtype=np.float64)
    for site in range(n):
        profile[site, 0] = trace_pauli(density, "X", site).real
        profile[site, 1] = trace_pauli(density, "Y", site).real
        profile[site, 2] = trace_pauli(density, "Z", site).real
    return profile


def _reindexed(operator: np.ndarray, row_flip: int, col_flip: int) -> np.ndarray:
    size = operator.shape[0]
    idx = np.arange(size)
    return operator[np.ix_(idx ^ row_flip, idx ^ col_flip)]


def walsh_degree_correlations(density: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed-basis local degree profile and connected correlations.

    In the X eigenbasis that defines fixed-basis Walshness, the site degree is
    d_i = r_i + q_i. For a Hermitian operator,

        E[d_i] = 1 - <X_i>_{σ}, σ = ρ²/Tr(ρ²)
        E[r_i r_j] = E[q_i q_j] = Tr(ρ² Π_i Π_j) / Tr(ρ²)
        E[r_i q_j] = Tr(ρ Π_i ρ Π_j) / Tr(ρ²)

    with Π = (I-X)/2. The returned correlation is the connected covariance of
    d_i and d_j.
    """
    if abs(np.max(np.abs(density - np.conjugate(density).T))) > 1e-8:
        raise ValueError("degree correlations expect a Hermitian operator")
    n = int(np.log2(density.shape[0]))
    squared = density @ density
    norm = float(np.trace(squared).real)
    x2 = np.array([trace_pauli(squared, "X", site).real / norm for site in range(n)])
    mean = 1.0 - x2
    # Two-point Pauli expectations of σ and the cross term Tr(ρ X_i ρ X_j)/norm.
    xx = np.zeros((n, n), dtype=np.float64)
    cross = np.zeros((n, n), dtype=np.float64)
    size = density.shape[0]
    idx = np.arange(size)
    for j in range(n):
        col_shift = _reindexed(density, 0, 1 << j)
        for i in range(n):
            mask = (1 << i) ^ (1 << j) if i != j else 0
            if i == j:
                xx[i, j] = 1.0  # <X_i²>_{σ} = 1
            else:
                xx[i, j] = float(np.sum(squared[idx, idx ^ (1 << i) ^ (1 << j)]).real / norm)
            shifted = col_shift[idx ^ (1 << i)]
            cross[i, j] = float(np.sum(density * shifted).real / norm)
    # E[r_i r_j] = (1 - <Xi> - <Xj> + <Xi Xj>)/4
    # E[r_i q_j] = (1 - <Xi> - <Xj> + cross_ij)/4
    # For i==j, r_i² = r_i, so E[r_i r_i] = E[r_i] = (1-<Xi>)/2, and the
    # four-term expansion still holds because <Xi²>=1:
    # (1 - 2<Xi> + 1)/4 = (1-<Xi>)/2. Good.
    profile_rr = np.zeros((n, n), dtype=np.float64)
    profile_rq = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            profile_rr[i, j] = (1.0 - x2[i] - x2[j] + xx[i, j]) / 4.0
            profile_rq[i, j] = (1.0 - x2[i] - x2[j] + cross[i, j]) / 4.0
    # E[d_i d_j] = 2 E[r_i r_j] + 2 E[r_i q_j] for the Hermitian identities
    # E[q_i q_j]=E[r_i r_j] and E[q_i r_j]=E[r_i q_j].
    raw = 2.0 * profile_rr + 2.0 * profile_rq
    connected = raw - np.outer(mean, mean)
    return {
        "degree_mean": mean,
        "degree_correlation": connected,
        "degree_second": raw,
    }


def operator_schmidt(operator: np.ndarray, n_left: int | None = None) -> tuple[np.ndarray, float]:
    """Half-chain operator Schmidt values and OSEE in bits.

    The vectorization groups ket and bra of the same site, local index
    s_i + 2 t_i, which is the grouping in the run plan. Singular values are
    normalized so that sum σ² = 1.
    """
    n = int(np.log2(operator.shape[0]))
    if n_left is None:
        n_left = n // 2
    if not 0 < n_left < n:
        raise ValueError("cut must separate the chain")
    size = 1 << n
    sites = np.arange(size)
    ket = sites[:, None]
    bra = sites[None, :]
    grouped = np.zeros((size, size), dtype=np.int64)
    for site in range(n):
        local = ((ket >> site) & 1) + 2 * ((bra >> site) & 1)
        grouped += local << (2 * site)
    flat = np.zeros(1 << (2 * n), dtype=np.result_type(operator, np.float64))
    flat[grouped.ravel()] = np.asarray(operator).ravel()
    n_right = n - n_left
    matrix = flat.reshape((1 << (2 * n_right), 1 << (2 * n_left)))
    singular = np.linalg.svd(matrix, compute_uv=False)
    weights = singular**2
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("operator is zero")
    weights /= total
    positive = weights > 0.0
    entropy = float(-(weights[positive] * np.log2(weights[positive])).sum())
    return weights, entropy


def state_half_entropy(state: np.ndarray, n_left: int | None = None) -> tuple[np.ndarray, float]:
    """Half-chain von Neumann entropy of a pure state, in bits."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    n = int(np.log2(state.size))
    if n_left is None:
        n_left = n // 2
    n_right = n - n_left
    # Low bits are the left sites.
    matrix = state.reshape((1 << n_right, 1 << n_left))
    singular = np.linalg.svd(matrix, compute_uv=False)
    weights = np.abs(singular) ** 2
    weights = weights / float(weights.sum())
    positive = weights > 1e-16
    entropy = float(-(weights[positive] * np.log2(weights[positive])).sum())
    return weights, entropy


def parity_expectation(state: np.ndarray) -> float:
    """<prod_i X_i> for a pure state."""
    state = np.asarray(state)
    flipped = state[::-1] if state.size == (1 << int(np.log2(state.size))) else None
    # Reversal flips all bits only when site 0 is LSB and the array is ordered
    # that way: index XOR (2^n-1) is exactly reversal. Yes.
    return float(np.vdot(state, state[::-1]).real)


def total_sz_expectation(state: np.ndarray) -> float:
    """<sum_i Z_i>/2, the usual total S^z."""
    state = np.asarray(state)
    n = int(np.log2(state.size))
    probs = np.abs(state) ** 2
    down = hamming_weights(n).astype(np.float64)
    magnetization = (n - 2.0 * down) / 2.0
    return float(np.dot(probs, magnetization))
