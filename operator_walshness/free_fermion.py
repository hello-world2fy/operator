"""Open-boundary transverse-field Ising model via free fermions.

The Hamiltonian

    H = -sum_i Z_i Z_{i+1} - g sum_i X_i

is the Majorana quadratic form (i/4) γ^T h γ, with

    X_j = i γ_{2j} γ_{2j+1},
    Z_j Z_{j+1} = i γ_{2j+1} γ_{2j+2}.

The fixed-basis generating function is the trace of four Gaussian operators.
On the positive real axis that trace is

    log Tr(e^{-βH} e^{-λ S} e^{-βH} e^{-λ S})
        = β sum_{ε>0} ε + (1/2) log det(W Mid Schur),

with the large thermal scales removed by a Schur complement, so the matrices
stay O(1) down to β = 16. S = sum X_j and λ = log(z)/2.

The square root that produces this determinant is branched. Along the unit
circle the branch is fixed by unwrapping the determinant phase on a grid fine
enough that the resulting polynomial reproduces the trusted real-axis moment.
Periodic boundaries are a different fermion-parity problem and are not claimed
by this module.
"""

from __future__ import annotations

import numpy as np

from .walsh import check_n


def _coupling(n: int, g: float) -> np.ndarray:
    matrix = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for site in range(n):
        matrix[2 * site, 2 * site + 1] = -2.0 * g
        matrix[2 * site + 1, 2 * site] = 2.0 * g
    for site in range(n - 1):
        left, right = 2 * site + 1, 2 * site + 2
        matrix[left, right] -= 2.0
        matrix[right, left] += 2.0
    return matrix


def _x_coupling(n: int) -> np.ndarray:
    matrix = np.zeros((2 * n, 2 * n), dtype=np.float64)
    for site in range(n):
        matrix[2 * site, 2 * site + 1] = 2.0
        matrix[2 * site + 1, 2 * site] = -2.0
    return matrix


def _apply_r2(vectors: np.ndarray, lam: complex) -> np.ndarray:
    """Apply exp(i λ h_S). Each site is a 2×2 block of the X coupling."""
    hyperbolic_cos = np.cosh(2.0 * lam)
    hyperbolic_sin = np.sinh(2.0 * lam)
    even = np.asarray(vectors[0::2], dtype=np.complex128)
    odd = np.asarray(vectors[1::2], dtype=np.complex128)
    out = np.empty(vectors.shape, dtype=np.complex128)
    out[0::2] = hyperbolic_cos * even + 1j * hyperbolic_sin * odd
    out[1::2] = -1j * hyperbolic_sin * even + hyperbolic_cos * odd
    return out


def _log_partition_from_energies(energies: np.ndarray, beta: float) -> float:
    if float(beta) == 0.0:
        return float(energies.size * np.log(2.0))
    return float(np.sum(np.log(2.0 * np.cosh(0.5 * float(beta) * energies))))


class OpenBoundaryTFIM:
    """Diagonalize the Majorana coupling once and reuse it for every temperature."""

    def __init__(self, n: int, g: float):
        self.n = check_n(n)
        self.g = float(g)
        values, vectors = np.linalg.eigh(1j * _coupling(self.n, self.g))
        # Ferromagnetic TFIM has a mode that is exponentially small in n. Split
        # the spectrum in half instead of thresholding it away.
        order = np.argsort(values)
        upper = order[self.n :]
        lower = order[: self.n]
        if float(values[upper].min()) < -1e-6:
            raise RuntimeError("Majorana spectrum is not paired into nonnegative modes")
        order = np.concatenate([upper, lower])
        self.eigenvalues = values[order]
        self.vectors = vectors[:, order].astype(np.complex128)
        self.energies = np.maximum(self.eigenvalues[: self.n].real, 0.0)
        self._negative_energies = np.maximum((-self.eigenvalues[self.n :]).real, 0.0)

    def log_partition(self, beta: float) -> float:
        return _log_partition_from_energies(self.energies, beta)

    def magnetization(self, beta: float) -> np.ndarray:
        """Gibbs expectations of (X, Y, Z). Y and Z vanish for this Hamiltonian."""
        profile = np.zeros((self.n, 3), dtype=np.float64)
        if float(beta) == 0.0:
            return profile
        weights = np.tanh(-0.5 * float(beta) * self.eigenvalues)
        even = self.vectors[0::2]
        odd = self.vectors[1::2]
        profile[:, 0] = np.sum(even * weights * np.conjugate(odd), axis=1).imag
        return profile

    def log_generating_abs(self, beta: float, alpha: float) -> float:
        """log |G(e^α)|. G itself is positive on this ray."""
        log_trace = self._log_trace_modulus(float(beta), 0.5 * float(alpha))
        return float(self.n * float(alpha) + log_trace - self.log_partition(2.0 * float(beta)))

    def fixed_walshness(self, beta: float, alpha: float) -> float:
        alpha = float(alpha)
        beta = float(beta)
        if alpha == 0.0:
            return float(np.sum(1.0 - self.magnetization(2.0 * beta)[:, 0]))
        return self.log_generating_abs(beta, alpha) / alpha

    def intrinsic_alpha0(self, beta: float) -> float:
        """Exact minimum at α→0. For this model it coincides with the fixed-X value."""
        return float(np.sum(1.0 - np.abs(self.magnetization(2.0 * float(beta))[:, 0])))

    def degree_spectrum(self, beta: float) -> np.ndarray:
        """Degree distribution from G on the unit circle.

        High powers e^{αk} amplify the roundoff in the far tail, so Walshness at
        α>0 is taken from the real-axis trace. Acceptance here is the α→0 mean,
        which is fixed by the local magnetizations and is insensitive to that tail.
        """
        beta = float(beta)
        exact_mean = self.fixed_walshness(beta, 0.0)
        for oversample in (4, 8, 16):
            spectrum, imaginary, negative = self._spectrum_on_grid(beta, oversample)
            mean = float(np.dot(np.arange(spectrum.size), spectrum))
            mean_error = abs(mean - exact_mean) / max(1.0, abs(exact_mean))
            if imaginary < 1e-8 and negative > -1e-9 and mean_error < 1e-8:
                return spectrum
        raise RuntimeError(
            f"TFIM spectrum branch did not settle for n={self.n}, g={self.g}, beta={beta}"
        )

    def _factors(self, beta: float, lam: complex):
        dimension = self.n
        transformed = self.vectors.conj().T @ _apply_r2(self.vectors, lam)
        inverse = np.linalg.inv(transformed)
        positive_scale = np.exp(-beta * self.energies)
        negative_scale = np.exp(-beta * self._negative_energies)
        blocks = (
            transformed[:dimension, :dimension],
            transformed[:dimension, dimension:],
            transformed[dimension:, :dimension],
            transformed[dimension:, dimension:],
        )
        inverse_blocks = (
            inverse[:dimension, :dimension],
            inverse[:dimension, dimension:],
            inverse[dimension:, :dimension],
            inverse[dimension:, dimension:],
        )
        middle = inverse_blocks[3] + negative_scale[:, None] * blocks[3] * negative_scale[None, :]
        top = blocks[0] + positive_scale[:, None] * inverse_blocks[0] * positive_scale[None, :]
        left = blocks[1] * negative_scale[None, :] + positive_scale[:, None] * inverse_blocks[1]
        right_target = negative_scale[:, None] * blocks[2] + inverse_blocks[2] * positive_scale[None, :]
        schur = top - left @ np.linalg.solve(middle, right_target)
        return transformed, middle, schur

    def _log_trace_modulus(self, beta: float, lam: complex) -> float:
        transformed, middle, schur = self._factors(beta, lam)
        modulus = 0.0
        for matrix in (transformed, middle, schur):
            _sign, log_det = np.linalg.slogdet(matrix)
            modulus += float(np.real(log_det))
        return float(beta * np.sum(self.energies) + 0.5 * modulus)

    def _phase_and_modulus(self, beta: float, lam: complex) -> tuple[float, float]:
        transformed, middle, schur = self._factors(beta, lam)
        sign = 1.0 + 0.0j
        modulus = 0.0
        for matrix in (transformed, middle, schur):
            this_sign, log_det = np.linalg.slogdet(matrix)
            sign *= this_sign
            modulus += float(np.real(log_det))
        return float(np.angle(sign)), modulus

    def _spectrum_on_grid(self, beta: float, oversample: int) -> tuple[np.ndarray, float, float]:
        length = 2 * self.n + 1
        grid = oversample * length
        angles = np.linspace(0.0, -2.0 * np.pi, grid, endpoint=False)
        phases = np.empty(grid, dtype=np.float64)
        moduli = np.empty(grid, dtype=np.float64)
        for index, angle in enumerate(angles):
            phases[index], moduli[index] = self._phase_and_modulus(beta, 0.5j * angle)
        half_phase = 0.5 * np.unwrap(phases)
        log_denominator = self.log_partition(2.0 * beta)
        shift = beta * float(np.sum(self.energies))
        samples = np.empty(length, dtype=np.complex128)
        for node in range(length):
            index = node * oversample
            angle = angles[index]
            log_abs = self.n * (1j * angle) + shift + 0.5 * moduli[index] - log_denominator
            samples[node] = np.exp(log_abs.real + 1j * (log_abs.imag + half_phase[index]))
        transformed = np.fft.ifft(samples)
        imaginary = float(np.max(np.abs(transformed.imag)))
        raw = transformed.real
        negative = float(np.sum(raw[raw < 0.0]))
        spectrum = np.clip(raw, 0.0, None)
        total = float(spectrum.sum())
        if total <= 0.0 or abs(total - 1.0) > 1e-4:
            raise RuntimeError(f"TFIM spectrum normalized to {total}")
        return spectrum / total, imaginary, negative


def quasiparticle_energies(n: int, g: float) -> np.ndarray:
    return OpenBoundaryTFIM(n, g).energies.copy()


def log_partition(n: int, g: float, beta: float) -> float:
    return OpenBoundaryTFIM(n, g).log_partition(beta)


def fixed_walshness(n: int, g: float, beta: float, alpha: float) -> float:
    return OpenBoundaryTFIM(n, g).fixed_walshness(beta, alpha)


def degree_spectrum(n: int, g: float, beta: float) -> np.ndarray:
    return OpenBoundaryTFIM(n, g).degree_spectrum(beta)


def local_magnetization(n: int, g: float, beta: float) -> np.ndarray:
    return OpenBoundaryTFIM(n, g).magnetization(beta)
