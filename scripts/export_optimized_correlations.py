"""Local Walsh-degree correlations in each saved physical basis.

The stored axes are the quantization directions n_i. Rotating every n_i onto
+Z makes the computational Hamming weight of the matrix elements the Walsh
degree along those axes. The resulting correlations are the connected
covariances of d_i = r_i + q_i. Numerical minima remain upper bounds; this
script only records the spatial structure of the basis that was saved.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize
from operator_walshness.observables import degree_correlations_from_x_matrix
from operator_walshness.walsh import (
    align_direction_to_z,
    frobenius_norm_squared,
    gibbs_density,
    hamming_weights,
    rotate_operator,
)


def _moment(rotated: np.ndarray, alpha: float) -> float:
    n = int(np.log2(rotated.shape[0]))
    if alpha == 0.0:
        weights = np.abs(rotated) ** 2
        weights /= float(weights.sum())
        bits = hamming_weights(n).astype(np.float64)
        return float(bits @ weights.sum(axis=1) + bits @ weights.sum(axis=0))
    weights = np.exp(0.5 * alpha * hamming_weights(n).astype(np.float64))
    scaled = weights[:, None] * rotated * weights[None, :]
    return float(np.vdot(scaled, scaled).real / frobenius_norm_squared(rotated))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results" / "round1" / "basis"
    destination = root / "results" / "round1" / "spatial"
    destination.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple, np.ndarray] = {}
    written = 0
    skipped = 0
    worst = 0.0
    for path in sorted(source.glob("*.npz")):
        name = path.name
        if not any(token in name for token in ("_n6_", "_n8_", "_n10_")):
            continue
        out = destination / name
        if out.exists():
            skipped += 1
            continue
        stored = np.load(path)
        if "axes_best" not in stored.files:
            continue
        n = int(stored["n"])
        beta = float(stored["beta"])
        alpha = float(stored["alpha"])
        params = json.loads(str(stored["params_json"]))
        key = (str(stored["model"]), n, str(stored["boundary"]), stored["params_json"].tobytes(), beta)
        if key not in cache:
            evals, evecs, _ = diagonalize(str(stored["model"]), n, str(stored["boundary"]), params)
            cache[key] = gibbs_density(evals, evecs, beta)[0]
        axes = np.asarray(stored["axes_best"], dtype=np.float64)
        unitaries = [align_direction_to_z(axes[site]) for site in range(n)]
        rotated = rotate_operator(cache[key], unitaries)
        correlations = degree_correlations_from_x_matrix(rotated)
        moment = _moment(rotated, alpha)
        reference = float(stored["W_upper"])
        if alpha == 0.0:
            disagreement = abs(moment - reference)
        else:
            disagreement = abs(np.log(moment) / alpha - reference)
        worst = max(worst, disagreement)
        np.savez_compressed(
            out,
            model=str(stored["model"]),
            n=n,
            boundary=str(stored["boundary"]),
            params_json=str(stored["params_json"]),
            beta=beta,
            alpha=alpha,
            degree_mean=correlations["degree_mean"],
            degree_correlation=correlations["degree_correlation"],
            degree_second=correlations["degree_second"],
            moment_disagreement=disagreement,
            W_upper=reference,
            source=name,
        )
        written += 1
        if written % 50 == 0:
            print(f"wrote {written} worst_disagreement {worst:.3e}", flush=True)
    print(f"done wrote {written} skipped {skipped} worst_disagreement {worst:.3e}", flush=True)


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
