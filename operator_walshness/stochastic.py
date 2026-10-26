"""Matrix-free Hutchinson estimate of fixed-basis Operator Walshness.

For O = exp(-β H),

    G(e^α) = ||A^{1/2} O A^{1/2}||_F^2 / ||O||_F^2
           = E ||A^{1/2} O A^{1/2} v||^2 / E ||O v||^2

with real Rademacher probes. The same probes form the ratio, so the estimate
is consistent but not unbiased; the reported error is the standard error of
the ratio across independent batches. This estimates a few values of W_α. It
does not produce the degree distribution P(k).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from .walsh import check_n


def _sqrt_factor(alpha: float) -> np.ndarray:
    scale = np.exp(0.5 * float(alpha))
    plus = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    minus = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    return plus + scale * minus


def apply_sqrt_a(state: np.ndarray, alpha: float, n: int) -> np.ndarray:
    """Apply ⊗_i A_i(α)^{1/2} with A the fixed-X Walsh factor."""
    factor = _sqrt_factor(alpha)
    out = np.asarray(state, dtype=np.float64).reshape([2] * n)
    for site in range(n):
        out = np.moveaxis(out, site, 0)
        out = np.tensordot(factor, out, axes=(1, 0))
        out = np.moveaxis(out, 0, site)
    return out.reshape(-1)


def estimate_walshness(
    hamiltonian: np.ndarray,
    beta: float,
    alpha: float,
    probes: int,
    seed: int,
    batches: int = 4,
) -> dict[str, float]:
    """Estimate W_α^{op,Z}(e^{-βH}). alpha=0 is not estimated here."""
    if probes < batches:
        raise ValueError("need at least one probe per batch")
    n = check_n(int(np.log2(hamiltonian.shape[0])))
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0.0:
        raise ValueError("stochastic estimate is for alpha>0; alpha=0 is a magnetization")
    rng = np.random.default_rng(seed)
    per_batch = probes // batches
    numerators = []
    denominators = []
    scaled = -beta * np.asarray(hamiltonian, dtype=np.float64)
    for _ in range(batches):
        num = 0.0
        den = 0.0
        for _probe in range(per_batch):
            probe = rng.choice(np.array([-1.0, 1.0]), size=1 << n)
            evolved = expm_multiply(scaled, probe, traceA=np.trace(scaled))
            den += float(np.dot(evolved, evolved))
            twisted = apply_sqrt_a(probe, alpha, n)
            evolved_twist = expm_multiply(scaled, twisted, traceA=np.trace(scaled))
            finished = apply_sqrt_a(evolved_twist, alpha, n)
            num += float(np.dot(finished, finished))
        numerators.append(num)
        denominators.append(den)
    ratios = np.array(numerators) / np.array(denominators)
    # The reported value pools every probe. The standard error is across batches.
    ratio = float(np.sum(numerators) / np.sum(denominators))
    stderr = float(np.std(ratios, ddof=1) / np.sqrt(batches)) if batches > 1 else 0.0
    return {
        "W": float(np.log(ratio) / alpha),
        "moment": ratio,
        "moment_stderr": stderr,
        "W_stderr": float(stderr / (abs(ratio) * alpha)),
        "probes": int(per_batch * batches),
        "batches": int(batches),
    }
