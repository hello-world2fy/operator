"""Fixed-basis Operator Walshness.

Conventions, all equivalent for the probabilities:

* vectorization |O>> = sum_{s,t} O(s,t) |s>⊗|t|
* unitary Hadamard matrix H = H_1^{⊗n}, H_1 = [[1,1],[1,-1]]/sqrt(2)
* p(r,q) = |(H O H)_{r,q}|^2 / Tr(O† O)
* degree k = |r| + |q|
* W_α = (1/α) log sum_k P(k) e^{α k} for α>0, and the mean degree at α=0
* density w = W / (2n)

The product-operator representation uses

    A_i = ((1+e^α)/2) I - ((e^α-1)/2) n_i·σ_i

and W from Tr(O† A O A) / Tr(O† O). The fixed computational-basis definition
corresponds to n_i = x-hat for every site, because H O H reads matrix elements
in the X eigenbasis and the low-degree pole is the +1 eigenstate of X.
"""

from __future__ import annotations

import numpy as np

from .hamiltonians import X, Y, Z, check_n

_I2 = np.eye(2, dtype=np.complex128)
_X = X.astype(np.complex128)
_Y = Y.astype(np.complex128)
_Z = Z.astype(np.complex128)


def hamming_weights(n: int) -> np.ndarray:
    n = check_n(n)
    return np.bitwise_count(np.arange(1 << n, dtype=np.uint32)).astype(np.int16)


def fwht(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unnormalized fast Walsh-Hadamard transform along one axis.

    The butterfly implements F_{r,s} = (-1)^{r·s}. The unitary Hadamard is
    F / 2^{n/2}.
    """
    x = np.array(values, copy=True)
    dtype = np.complex128 if np.iscomplexobj(x) else np.float64
    x = np.asarray(x, dtype=dtype)
    x = np.moveaxis(x, axis, -1)
    length = x.shape[-1]
    if length <= 0 or length & (length - 1):
        raise ValueError("FWHT axis length must be a positive power of two")
    h = 1
    while h < length:
        x = x.reshape(*x.shape[:-1], length // (2 * h), 2, h)
        left = np.array(x[..., 0, :], copy=True)
        right = x[..., 1, :]
        x[..., 0, :] = left + right
        x[..., 1, :] = left - right
        x = x.reshape(*x.shape[:-3], length)
        h *= 2
    return np.moveaxis(x, -1, axis)


def x_basis_matrix(operator: np.ndarray) -> np.ndarray:
    """Return the unitary-Hadamard conjugate H O H."""
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be a square matrix")
    n = int(np.log2(operator.shape[0]))
    if (1 << n) != operator.shape[0]:
        raise ValueError("operator dimension must be a power of two")
    transformed = fwht(fwht(operator, axis=0), axis=1)
    return transformed / operator.shape[0]


def frobenius_norm_squared(operator: np.ndarray) -> float:
    return float(np.vdot(operator, operator).real)


def degree_histogram(x_matrix: np.ndarray, n: int | None = None) -> tuple[np.ndarray, float]:
    """Bin |(H O H)_{r,q}|^2 by |r|+|q|.

    Returns the unnormalized masses and their sum. The sum equals ||O||_F^2
    up to roundoff when ``x_matrix`` is exactly H O H.
    """
    if n is None:
        n = int(np.log2(x_matrix.shape[0]))
    n = check_n(n)
    hw = hamming_weights(n).astype(np.int32)
    masses = np.zeros(2 * n + 1, dtype=np.float64)
    weights = np.abs(x_matrix) ** 2
    step = 256
    for start in range(0, weights.shape[0], step):
        block = weights[start : start + step]
        degree = hw[start : start + block.shape[0], None] + hw[None, :]
        masses += np.bincount(
            degree.ravel().astype(np.int64),
            weights=block.reshape(-1),
            minlength=2 * n + 1,
        )
    return masses, float(masses.sum())


def spectrum_from_masses(masses: np.ndarray) -> np.ndarray:
    total = float(masses.sum())
    if total <= 0.0:
        raise ValueError("operator is zero; Walshness is undefined")
    spectrum = masses / total
    negative = float(np.min(spectrum))
    if negative < -1e-10:
        raise ValueError(f"Walsh spectrum has a negative bin {negative}")
    return np.clip(spectrum, 0.0, None)


def walshness_from_spectrum(spectrum: np.ndarray, alpha: float) -> float:
    """W_α from a normalized degree distribution. α=0 returns the mean degree."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    spectrum = np.asarray(spectrum, dtype=np.float64)
    degrees = np.arange(spectrum.size, dtype=np.float64)
    if alpha == 0.0:
        return float(np.dot(degrees, spectrum))
    positive = spectrum > 0.0
    scores = np.log(spectrum[positive]) + alpha * degrees[positive]
    shift = float(np.max(scores))
    log_moment = shift + np.log(float(np.sum(np.exp(scores - shift))))
    return float(log_moment / alpha)


def spectrum_moments(spectrum: np.ndarray) -> dict[str, float]:
    degrees = np.arange(spectrum.size, dtype=np.float64)
    mean = float(np.dot(degrees, spectrum))
    second = float(np.dot(degrees**2, spectrum))
    variance = second - mean**2
    centered = degrees - mean
    third = float(np.dot(centered**3, spectrum))
    return {
        "mean_degree": mean,
        "variance": variance,
        "second_moment": second,
        "third_central": third,
    }


def tail_degree(spectrum: np.ndarray, eps: float) -> int:
    """Smallest D such that sum_{k>D} P(k) <= eps."""
    eps = float(eps)
    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    tail_from = np.cumsum(spectrum[::-1])[::-1]
    # tail_from[d] = sum_{k>=d} P(k); the strict tail above D is tail_from[D+1].
    for degree in range(-1, spectrum.size):
        if degree + 1 >= spectrum.size:
            return int(spectrum.size - 1)
        if tail_from[degree + 1] <= eps + 1e-15:
            return int(max(degree, 0) if degree >= 0 else 0)
    return int(spectrum.size - 1)


def fixed_basis_spectrum(operator: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Full fixed-basis degree distribution of a dense operator."""
    n = int(np.log2(operator.shape[0]))
    masses, mass_sum = degree_histogram(x_basis_matrix(operator), n)
    fro = frobenius_norm_squared(operator)
    spectrum = spectrum_from_masses(masses)
    diagnostics = {
        "mass_sum": mass_sum,
        "frobenius_squared": fro,
        "mass_relative_error": abs(mass_sum - fro) / max(fro, 1e-300),
        "probability_defect": abs(float(spectrum.sum()) - 1.0),
        "min_bin": float(np.min(spectrum)),
    }
    return spectrum, diagnostics


def walshness(operator: np.ndarray, alpha: float) -> float:
    spectrum, _ = fixed_basis_spectrum(operator)
    return walshness_from_spectrum(spectrum, alpha)


def identity_walshness(n: int, alpha: float) -> float:
    """Analytic fixed-basis (and intrinsic) Walshness of any nonzero multiple of I."""
    n = check_n(n)
    alpha = float(alpha)
    if alpha == 0.0:
        return float(n)
    return float(n * np.log((1.0 + np.exp(2.0 * alpha)) / 2.0) / alpha)


def identity_spectrum(n: int) -> np.ndarray:
    n = check_n(n)
    spectrum = np.zeros(2 * n + 1, dtype=np.float64)
    # Only r=q contributes, with |r|+|q| = 2|r| and P(2m) = C(n,m)/2^n.
    choose = 1.0
    for m in range(n + 1):
        spectrum[2 * m] = choose / (1 << n)
        choose *= (n - m) / (m + 1)
    return spectrum


def product_state_flat_spectrum(n: int) -> np.ndarray:
    """Fixed-basis spectrum of |0...0><0...0|: 2n independent fair bits."""
    n = check_n(n)
    # Convolution of 2n Bernoulli(1/2) variables via the binomial theorem.
    spectrum = np.zeros(2 * n + 1, dtype=np.float64)
    choose = 1.0
    for k in range(2 * n + 1):
        spectrum[k] = choose / float(1 << (2 * n))
        choose *= (2 * n - k) / (k + 1)
    return spectrum


def local_factor(direction: np.ndarray, alpha: float) -> np.ndarray:
    """2×2 factor ((1+e^α)/2) I - ((e^α-1)/2) n·σ."""
    alpha = float(alpha)
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("Bloch direction must be nonzero")
    direction = direction / norm
    a = 0.5 * (1.0 + np.exp(alpha))
    b = 0.5 * (np.exp(alpha) - 1.0)
    nx, ny, nz = direction
    return a * _I2 - b * (nx * _X + ny * _Y + nz * _Z)


def _site_ket_axis(site: int, n: int) -> int:
    return n - 1 - site


def apply_local_ket(operator: np.ndarray, local: np.ndarray, site: int, n: int) -> np.ndarray:
    """Left-multiply by a single-site operator."""
    size = 1 << n
    tensor = np.reshape(operator, [2] * n + [size])
    axis = _site_ket_axis(site, n)
    tensor = np.moveaxis(tensor, axis, 0)
    tensor = np.tensordot(local, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, axis)
    return np.reshape(tensor, (size, size))


def apply_local_bra(operator: np.ndarray, local: np.ndarray, site: int, n: int) -> np.ndarray:
    """Right-multiply by a single-site operator."""
    size = 1 << n
    tensor = np.reshape(operator, [size] + [2] * n)
    axis = 1 + _site_ket_axis(site, n)
    tensor = np.moveaxis(tensor, axis, 0)
    tensor = np.tensordot(local, tensor, axes=([0], [0]))
    tensor = np.moveaxis(tensor, 0, axis)
    return np.reshape(tensor, (size, size))


def apply_product_operator(operator: np.ndarray, locals_: list[np.ndarray]) -> np.ndarray:
    n = len(locals_)
    out = operator
    for site, local in enumerate(locals_):
        out = apply_local_ket(out, local, site, n)
        out = apply_local_bra(out, local, site, n)
    return out


def moment(operator: np.ndarray, directions: np.ndarray, alpha: float) -> float:
    """Tr(O† A(n) O A(n)) / Tr(O† O), with one Bloch vector per site."""
    alpha = float(alpha)
    directions = np.asarray(directions, dtype=np.float64)
    n = directions.shape[0]
    if alpha == 0.0:
        return 1.0
    locals_ = [local_factor(directions[site], alpha) for site in range(n)]
    transformed = apply_product_operator(np.asarray(operator, dtype=np.complex128), locals_)
    numerator = np.vdot(operator, transformed).real
    denominator = frobenius_norm_squared(operator)
    if denominator <= 0.0:
        raise ValueError("operator is zero; Walshness is undefined")
    return float(numerator / denominator)


def walshness_from_directions(operator: np.ndarray, directions: np.ndarray, alpha: float) -> float:
    alpha = float(alpha)
    if alpha == 0.0:
        return mean_degree_from_directions(operator, directions)
    value = moment(operator, directions, alpha)
    if value <= 0.0:
        raise ValueError("moment must be positive")
    return float(np.log(value) / alpha)


def fixed_x_directions(n: int) -> np.ndarray:
    n = check_n(n)
    directions = np.zeros((n, 3), dtype=np.float64)
    directions[:, 0] = 1.0
    return directions


def align_direction_to_z(direction: np.ndarray) -> np.ndarray:
    """SU(2) u with u† (n·σ) u = Z. Columns are |+n> and |-n|."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    nx, ny, nz = direction
    theta = float(np.arccos(np.clip(nz, -1.0, 1.0)))
    phi = float(np.arctan2(ny, nx))
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    phase = np.exp(1j * phi)
    return np.array(
        [
            [cosine, -np.exp(-1j * phi) * sine],
            [phase * sine, cosine],
        ],
        dtype=np.complex128,
    )


def rotate_operator(operator: np.ndarray, unitaries: list[np.ndarray]) -> np.ndarray:
    """Return U† O U for U=⊗ u_i."""
    n = len(unitaries)
    out = np.asarray(operator, dtype=np.complex128)
    for site, unitary in enumerate(unitaries):
        out = apply_local_ket(out, unitary.conj().T, site, n)
        out = apply_local_bra(out, unitary, site, n)
    return out


def moment_by_rotation(operator: np.ndarray, directions: np.ndarray, alpha: float) -> float:
    """Same moment, evaluated after rotating every n_i onto +Z."""
    alpha = float(alpha)
    n = directions.shape[0]
    unitaries = [align_direction_to_z(directions[site]) for site in range(n)]
    rotated = rotate_operator(operator, unitaries)
    weights = np.exp(0.5 * alpha * hamming_weights(n).astype(np.float64))
    scaled = weights[:, None] * rotated * weights[None, :]
    numerator = float(np.vdot(scaled, scaled).real)
    denominator = frobenius_norm_squared(operator)
    return numerator / denominator


def state_degree_spectrum(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    n = int(np.log2(state.size))
    amplitudes = fwht(state) / np.sqrt(state.size)
    probabilities = np.abs(amplitudes) ** 2
    normal = float(probabilities.sum())
    if normal <= 0.0:
        raise ValueError("state is zero")
    probabilities /= normal
    return np.bincount(hamming_weights(n).astype(np.int64), weights=probabilities, minlength=n + 1)


def state_walshness(state: np.ndarray, alpha: float) -> float:
    return walshness_from_spectrum(state_degree_spectrum(state), alpha)


def trace_pauli(operator: np.ndarray, pauli: str, site: int) -> complex:
    size = operator.shape[0]
    idx = np.arange(size)
    if pauli == "X":
        return np.sum(operator[idx, idx ^ (1 << site)])
    if pauli == "Z":
        bit = (idx >> site) & 1
        return np.sum(np.diag(operator) * (1 - 2 * bit))
    if pauli == "Y":
        bit = (idx >> site) & 1
        phase = np.where(bit == 0, 1.0j, -1.0j)
        # <s|O Y|s> = O[s, s xor] * <s xor| Y |s>
        # Y|s> = phase(bit_s) |s xor>, with phase +i if bit 0 else -i.
        # Trace sums <s| O Y |s> = O[s, s xor] * phase(bit of s).
        return np.sum(operator[idx, idx ^ (1 << site)] * phase)
    raise ValueError(pauli)


def normalized_expectations(operator: np.ndarray) -> tuple[np.ndarray, float]:
    """Single-site Bloch vector of σ = O / Tr(O), requiring a nonzero trace.

    This is the ordinary thermal magnetization when ``operator`` is a density
    matrix. It is not the ρ² magnetization used by the α→0 identity.
    """
    n = int(np.log2(operator.shape[0]))
    trace = np.trace(operator)
    if abs(trace) == 0.0:
        raise ValueError("operator has zero trace")
    bloch = np.zeros((n, 3), dtype=np.float64)
    for site in range(n):
        bloch[site, 0] = (trace_pauli(operator, "X", site) / trace).real
        bloch[site, 1] = (trace_pauli(operator, "Y", site) / trace).real
        bloch[site, 2] = (trace_pauli(operator, "Z", site) / trace).real
    return bloch, float(trace.real)


def rho_squared_bloch(operator: np.ndarray) -> tuple[np.ndarray, float]:
    """Bloch vector of σ = O† O / Tr(O† O) for Hermitian O, via O² when Hermitian.

    The α→0 derivation uses Tr(ρ² σ_i)/Tr(ρ²). For a general nonzero operator
    the same traces are Tr(O† O σ_i)/Tr(O† O) only after the row/column
    identification, which needs Hermiticity. Callers pass a Hermitian operator.
    """
    squared = operator @ operator
    trace = np.trace(squared).real
    if trace <= 0.0:
        raise ValueError("operator is zero")
    n = int(np.log2(operator.shape[0]))
    bloch = np.zeros((n, 3), dtype=np.float64)
    for site in range(n):
        bloch[site, 0] = (trace_pauli(squared, "X", site) / trace).real
        bloch[site, 1] = (trace_pauli(squared, "Y", site) / trace).real
        bloch[site, 2] = (trace_pauli(squared, "Z", site) / trace).real
    return bloch, float(trace)


def mean_degree_from_directions(operator: np.ndarray, directions: np.ndarray) -> float:
    """α→0 Walshness at fixed Bloch axes: sum_i (1 - <n_i·σ_i>_{O²})."""
    bloch, _ = rho_squared_bloch(operator)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projections = np.sum(bloch * directions, axis=1)
    return float(np.sum(1.0 - projections))


def intrinsic_alpha0(operator: np.ndarray) -> tuple[float, np.ndarray]:
    """Globally minimal α→0 Operator Walshness and the aligned axes.

    The minimum is sum_i (1-|m_i|), attained by aligning n_i with the Bloch
    vector of O². Vanishing Bloch vectors are reported as +x, an arbitrary
    minimum direction.
    """
    bloch, _ = rho_squared_bloch(operator)
    lengths = np.linalg.norm(bloch, axis=1)
    axes = np.zeros_like(bloch)
    for site, length in enumerate(lengths):
        if length < 1e-14:
            axes[site, 0] = 1.0
        else:
            axes[site] = bloch[site] / length
    return float(np.sum(1.0 - lengths)), axes


def convolution_spectrum(factors: list[np.ndarray]) -> np.ndarray:
    """Exact degree distribution of a product of operators on disjoint sites."""
    if not factors:
        raise ValueError("empty product")
    length = int(sum(item.size - 1 for item in factors) + 1)
    acc = np.array([1.0], dtype=np.float64)
    for factor in factors:
        acc = np.convolve(acc, np.asarray(factor, dtype=np.float64))
    if acc.size != length:
        raise RuntimeError("convolution length mismatch")
    total = float(acc.sum())
    return acc / total


def gibbs_density(evals: np.ndarray, evecs: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized Gibbs state and its eigenvalue probabilities.

    Eigenvectors are columns. β=0 returns the maximally mixed state.
    """
    evals = np.asarray(evals, dtype=np.float64)
    if float(beta) == 0.0:
        probabilities = np.full(evals.size, 1.0 / evals.size)
    else:
        logits = -float(beta) * evals
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        probabilities = weights / float(weights.sum())
    density = (evecs * probabilities) @ np.conjugate(evecs).T
    return np.real_if_close(density, tol=1000).astype(np.float64, copy=False), probabilities
