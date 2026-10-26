"""Exact large-n routes that do not build a density matrix.

Classical Ising uses a two-replica transfer matrix. Product models use the
fact that the degree distribution of a product operator is a convolution.
Transverse-field Ising uses a free-fermion trace, calibrated against exact
diagonalization in the tests.
"""

from __future__ import annotations

import numpy as np

from .hamiltonians import check_n


def _binomial_filter(n_free: int, offset: int, length: int) -> np.ndarray:
    """Distribution of offset + 2*Binomial(n_free, 1/2), as a vector of `length`."""
    out = np.zeros(length, dtype=np.float64)
    if n_free < 0:
        return out
    choose = 1.0
    scale = 0.5**n_free
    for j in range(n_free + 1):
        degree = offset + 2 * j
        if 0 <= degree < length:
            out[degree] += choose * scale
        choose *= (n_free - j) / (j + 1)
    return out


def classical_diagonal_spectrum_from_boltzmann(weights: np.ndarray) -> np.ndarray:
    """Operator Walsh spectrum of a Z-diagonal operator.

    ``weights[s]`` is the diagonal matrix element. Only the Hamming-weight
    distribution of its unitary Walsh transform is required: conditionally on
    a mask of weight w, |r|+|r XOR mask| = w + 2*Binomial(n-w, 1/2).
    """
    weights = np.asarray(weights, dtype=np.float64)
    n = int(np.log2(weights.size))
    if (1 << n) != weights.size:
        raise ValueError("Boltzmann vector length must be a power of two")
    from .walsh import fwht

    hat = fwht(weights) / np.sqrt(weights.size)
    mask_prob = np.abs(hat) ** 2
    mask_prob /= float(mask_prob.sum())
    mask_weight = np.bitwise_count(np.arange(weights.size, dtype=np.uint32))
    grouped = np.bincount(mask_weight, weights=mask_prob, minlength=n + 1)
    spectrum = np.zeros(2 * n + 1, dtype=np.float64)
    for weight, probability in enumerate(grouped):
        if probability == 0.0:
            continue
        spectrum += probability * _binomial_filter(n - weight, weight, 2 * n + 1)
    return spectrum


def classical_chain_moment(n: int, beta: float, alpha: float, boundary: str = "open") -> float:
    """G(e^α) for H=-sum Z_i Z_{i+1}, fixed computational basis."""
    return float(np.exp(_classical_chain_log_generating_value(n, beta, np.exp(alpha), boundary)).real)


def classical_chain_walshness(n: int, beta: float, alpha: float, boundary: str = "open") -> float:
    """Fixed-basis W_α. α=0 is recovered by a short finite difference of log G."""
    alpha = float(alpha)
    if alpha == 0.0:
        step = 1e-6
        return classical_chain_walshness(n, beta, step, boundary)
    return float(_classical_chain_log_generating_value(n, beta, np.exp(alpha), boundary).real / alpha)


def classical_staggered_z_walshness(n: int, beta: float, alpha: float, boundary: str = "open") -> float:
    """Walshness of classical ZZ in the staggered ±Z basis.

    Even sites use +Z and odd sites use -Z. The operator is diagonal, so the
    moment is an ordinary Ising expectation at inverse temperature 2β. The
    value is exact in that basis. It is the intrinsic minimum only when it
    equals the α→0 floor, which is n at every temperature.
    """
    alpha = float(alpha)
    if alpha == 0.0:
        step = 1e-6
        return classical_staggered_z_walshness(n, beta, step, boundary)
    log_moment = _staggered_z_log_generating_value(n, beta, np.exp(alpha), boundary)
    return float(log_moment.real / alpha)


def classical_staggered_z_spectrum(n: int, beta: float, boundary: str = "open") -> np.ndarray:
    """Degree distribution in the staggered ±Z basis, from Fourier inversion of G."""
    n = check_n(n)
    length = 2 * n + 1
    angles = -2.0 * np.pi * np.arange(length) / length
    samples = np.empty(length, dtype=np.complex128)
    for index, angle in enumerate(angles):
        log_value = _staggered_z_log_generating_value(n, beta, np.exp(1j * angle), boundary)
        samples[index] = np.exp(log_value)
    spectrum = np.fft.ifft(samples).real
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(spectrum.sum())
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError(f"staggered spectrum failed to normalize: sum={total}")
    return spectrum / total


def _staggered_z_log_generating_value(n: int, beta: float, z: complex, boundary: str) -> complex:
    """Log of E[z^{2 N_-}] for the 2β classical chain."""
    n = check_n(n)
    periodic = boundary.lower() in {"periodic", "pbc"}
    if periodic and n % 2:
        raise ValueError("a staggered ±Z texture does not close on an odd periodic chain")
    log_numerator = _staggered_z_log_partition(n, 2.0 * float(beta), complex(z), periodic)
    log_denominator = _staggered_z_log_partition(n, 2.0 * float(beta), 1.0 + 0.0j, periodic)
    return log_numerator - log_denominator


def _staggered_site_weight(site: int, z: complex) -> np.ndarray:
    # Bit 0 is the +1 eigenstate of Z. A +Z axis contributes z on bit 1;
    # a -Z axis contributes z on bit 0. The moment squares that factor.
    factor = z * z
    if site % 2 == 0:
        return np.array([1.0, factor], dtype=np.complex128)
    return np.array([factor, 1.0], dtype=np.complex128)


def _staggered_z_log_partition(n: int, beta2: float, z: complex, periodic: bool) -> complex:
    spins = np.array([1.0, -1.0])

    def transfer(site: int) -> np.ndarray:
        weight = _staggered_site_weight(site, z)
        matrix = np.zeros((2, 2), dtype=np.complex128)
        for left in range(2):
            for right in range(2):
                bond = np.exp(beta2 * spins[left] * spins[right])
                matrix[left, right] = bond * weight[right]
        return matrix

    if periodic:
        propagate = np.diag(_staggered_site_weight(0, z)).astype(np.complex128)
        propagate, log_scale = _rescale(propagate, 0.0)
        for site in range(1, n):
            propagate = transfer(site).T @ propagate
            propagate, log_scale = _rescale(propagate, log_scale)
        closing = np.zeros((2, 2), dtype=np.complex128)
        for first in range(2):
            for last in range(2):
                closing[last, first] = np.exp(beta2 * spins[first] * spins[last])
        propagate = closing.T @ propagate
        return np.log(np.trace(propagate)) + log_scale
    vector, log_scale = _rescale(_staggered_site_weight(0, z), 0.0)
    for site in range(1, n):
        vector = transfer(site).T @ vector
        vector, log_scale = _rescale(vector, log_scale)
    return np.log(np.sum(vector)) + log_scale


def classical_chain_spectrum(n: int, beta: float, boundary: str = "open") -> np.ndarray:
    """Full degree distribution from a discrete Fourier inversion of G(z)."""
    n = check_n(n)
    length = 2 * n + 1
    # Numpy's FFT uses e^{-2π i j k / N}. Sampling G at those roots makes ifft exact.
    angles = -2.0 * np.pi * np.arange(length) / length
    samples = np.empty(length, dtype=np.complex128)
    for index, angle in enumerate(angles):
        samples[index] = _classical_chain_generating_value(n, beta, np.exp(1j * angle), boundary)
    spectrum = np.fft.ifft(samples).real
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(spectrum.sum())
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError(f"classical spectrum failed to normalize: sum={total}")
    return spectrum / total


def _classical_chain_generating_value(n: int, beta: float, z: complex, boundary: str) -> complex:
    """Tr(Õ B(z) Õ B(z)) / Tr(Õ²) with Õ=exp(-βH) diagonal."""
    return np.exp(_classical_chain_log_generating_value(n, beta, z, boundary))


def _classical_chain_log_generating_value(n: int, beta: float, z: complex, boundary: str) -> complex:
    n = check_n(n)
    log_numerator = _replica_log_partition(n, beta, z, boundary)
    log_denominator = _replica_log_partition(n, beta, 1.0 + 0.0j, boundary)
    return log_numerator - log_denominator


def _rescale(vector: np.ndarray, log_scale: float) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0 or not np.isfinite(scale):
        return vector, -np.inf
    return vector / scale, log_scale + float(np.log(scale))


def _replica_log_partition(n: int, beta: float, z: complex, boundary: str) -> complex:
    """Complex log of the two-replica partition, stable at large n and β."""
    p = 0.5 * (1.0 + z)
    q = 0.5 * (1.0 - z)
    same = p * p
    diff = q * q
    site_weight = np.array([same, diff, diff, same], dtype=np.complex128)
    transfer = np.zeros((4, 4), dtype=np.complex128)
    spins = np.array([1.0, -1.0])
    for left in range(4):
        s_left, t_left = left % 2, left // 2
        for right in range(4):
            s_right, t_right = right % 2, right // 2
            bond = np.exp(beta * spins[s_left] * spins[s_right])
            bond *= np.exp(beta * spins[t_left] * spins[t_right])
            transfer[left, right] = bond * site_weight[right]
    if boundary.lower() in {"periodic", "pbc"}:
        total, log_scale = _periodic_replica(n, beta, transfer, site_weight)
        return np.log(total) + log_scale
    vector, log_scale = _rescale(site_weight.astype(np.complex128), 0.0)
    for _ in range(n - 1):
        vector = transfer.T @ vector
        vector, log_scale = _rescale(vector, log_scale)
    return np.log(np.sum(vector)) + log_scale


def _periodic_replica(n: int, beta: float, transfer: np.ndarray, site_weight: np.ndarray) -> tuple[complex, float]:
    """Periodic two-replica partition as (scaled trace, log of the scale)."""
    spins = np.array([1.0, -1.0])
    closing = np.zeros((4, 4), dtype=np.complex128)
    for first in range(4):
        s_first, t_first = first % 2, first // 2
        for last in range(4):
            s_last, t_last = last % 2, last // 2
            bond = np.exp(beta * spins[s_first] * spins[s_last])
            bond *= np.exp(beta * spins[t_first] * spins[t_last])
            closing[last, first] = bond
    propagate = np.diag(site_weight).astype(np.complex128)
    propagate, log_scale = _rescale(propagate, 0.0)
    for _ in range(n - 1):
        propagate = transfer.T @ propagate
        propagate, log_scale = _rescale(propagate, log_scale)
    propagate = closing.T @ propagate
    propagate, log_scale = _rescale(propagate, log_scale)
    return np.sum(np.diag(propagate)), log_scale


def classical_chain_boltzmann(n: int, beta: float, boundary: str = "open") -> np.ndarray:
    from .hamiltonians import classical_energies_zz

    return np.exp(-beta * classical_energies_zz(n, boundary))


def convolve_copies(spectrum: np.ndarray, copies: int) -> np.ndarray:
    if copies < 1:
        raise ValueError("copies must be positive")
    out = np.array([1.0], dtype=np.float64)
    factor = np.asarray(spectrum, dtype=np.float64)
    for _ in range(copies):
        out = np.convolve(out, factor)
    total = float(out.sum())
    return out / total


def product_x_spectrum(n: int, g: float, beta: float) -> np.ndarray:
    """Fixed-basis spectrum of ⊗ ρ_1 for H=-g sum X_i.

    The single-site Gibbs state is diagonal in X, so only degrees 0 and 2 occur.
    """
    n = check_n(n)
    field = float(g) * float(beta)
    plus = np.exp(field)
    minus = np.exp(-field)
    purity = plus**2 + minus**2
    local = np.array([plus**2 / purity, 0.0, minus**2 / purity], dtype=np.float64)
    return convolve_copies(local, n)


def ising_2d_boltzmann(lx: int, ly: int, beta: float) -> np.ndarray:
    """Open rectangle, site (x,y) = x + lx*y, bit x+lx*y."""
    n = lx * ly
    if n > 20:
        raise ValueError("direct Boltzmann vector is only for n<=20; use the cylinder transfer")
    idx = np.arange(1 << n)
    energy = np.zeros(idx.size, dtype=np.float64)
    spins = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    for y in range(ly):
        for x in range(lx):
            s = x + lx * y
            if x + 1 < lx:
                energy += -spins[:, s] * spins[:, s + 1]
            if y + 1 < ly:
                energy += -spins[:, s] * spins[:, s + lx]
    return np.exp(-beta * energy)


def cylinder_spectrum(length: int, width: int, beta: float, alpha_samples: int | None = None) -> np.ndarray:
    """Open-length, periodic-circumference classical Ising operator spectrum.

    Two replicas make the row space 4^width. This is the geometry control, not
    a substitute for the 1d census.
    """
    if width > 6:
        raise ValueError("cylinder width above 6 is a dense transfer larger than this helper targets")
    n = length * width
    count = 2 * n + 1 if alpha_samples is None else alpha_samples
    if count < 2 * n + 1:
        raise ValueError("need at least 2n+1 samples to invert the degree polynomial")
    angles = -2.0 * np.pi * np.arange(count) / count
    samples = np.empty(count, dtype=np.complex128)
    for index, angle in enumerate(angles):
        samples[index] = _cylinder_generating_value(length, width, beta, np.exp(1j * angle))
    spectrum = np.fft.ifft(samples).real
    if count != 2 * n + 1:
        spectrum = spectrum[: 2 * n + 1]
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(spectrum.sum())
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"cylinder spectrum normalization {total}")
    return spectrum / total


def _cylinder_generating_value(length: int, width: int, beta: float, z: complex) -> complex:
    numerator = _cylinder_partition(length, width, beta, z)
    denominator = _cylinder_partition(length, width, beta, 1.0 + 0.0j)
    return numerator / denominator


def _cylinder_partition(length: int, width: int, beta: float, z: complex) -> complex:
    p = 0.5 * (1.0 + z)
    q = 0.5 * (1.0 - z)
    same = p * p
    diff = q * q
    row = 1 << width
    dim = row * row
    # Encode a row of replica s in the low `width` bits and replica t in the high bits.
    transfer = np.zeros((dim, dim), dtype=np.complex128)
    spins = lambda bits, i: 1.0 if ((bits >> i) & 1) == 0 else -1.0
    for left in range(dim):
        s_left, t_left = left % row, left // row
        # Vertical bonds and on-site B weights live on the destination row,
        # matching the 1d convention of charging the right site.
        for right in range(dim):
            s_right, t_right = right % row, right // row
            weight = 1.0 + 0.0j
            for i in range(width):
                weight *= np.exp(beta * spins(s_left, i) * spins(s_right, i))
                weight *= np.exp(beta * spins(t_left, i) * spins(t_right, i))
                equal_s = ((s_right >> i) & 1) == ((t_right >> i) & 1)
                weight *= same if equal_s else diff
                # Circumference bond inside the row.
                j = (i + 1) % width
                weight *= np.exp(beta * spins(s_right, i) * spins(s_right, j))
                weight *= np.exp(beta * spins(t_right, i) * spins(t_right, j))
            transfer[left, right] = weight
    # First row has vertical weights from a fictitious empty left neighbor.
    # Rebuild the first-row vector without incoming vertical bonds, keeping
    # circumference bonds and B weights.
    vector = np.zeros(dim, dtype=np.complex128)
    for right in range(dim):
        s_right, t_right = right % row, right // row
        weight = 1.0 + 0.0j
        for i in range(width):
            equal_s = ((s_right >> i) & 1) == ((t_right >> i) & 1)
            weight *= same if equal_s else diff
            j = (i + 1) % width
            weight *= np.exp(beta * spins(s_right, i) * spins(s_right, j))
            weight *= np.exp(beta * spins(t_right, i) * spins(t_right, j))
        vector[right] = weight
    for _ in range(length - 1):
        vector = transfer.T @ vector
    return np.sum(vector)
