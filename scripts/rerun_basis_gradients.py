"""Re-polish basis points whose gradient missed the tolerance.

The new value is still an upper bound. It replaces the saved file only when
it is lower than the previous upper bound.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from operator_walshness.exact import diagonalize
from operator_walshness.grid import case_name
from operator_walshness.optimize import optimize_walshness
from operator_walshness.walsh import gibbs_density
from run_basis_opt import _code_hash
from run_basis_other_alphas import _alpha1_axes

TASKS = (
    ("mixed_field_ising", 10, "open", {"g": 1.2, "h": 0.5}, 1.0, 2.0),
    ("mixed_field_ising", 10, "open", {"g": 1.05, "h": 0.5}, 1.0, 2.0),
    ("mixed_field_ising", 10, "open", {"g": 0.8, "h": 0.2}, 1.0, 2.0),
    ("mixed_field_ising", 10, "open", {"g": 0.8, "h": 0.2}, 2.0, 2.0),
    ("xxz", 10, "open", {"delta": 0.5}, 2.0, 2.0),
)


def _one(task: tuple) -> str:
    model, n, boundary, params, beta, alpha, device = task
    out_dir = str(Path(__file__).resolve().parents[1] / "results" / "round1" / "basis")
    name = case_name(model, n, boundary, params) + f"_b{beta:g}_a{alpha:g}_s0"
    destination = Path(out_dir) / f"{name}.npz"
    previous = float(np.load(destination)["W_upper"]) if destination.exists() else np.inf
    started = time.time()
    evals, evecs, _ = diagonalize(model, n, boundary, params)
    density, _ = gibbs_density(evals, evecs, beta)
    neighbor = _alpha1_axes(model, n, boundary, params, beta, out_dir)
    result = optimize_walshness(
        density,
        alpha,
        restarts=16,
        seed=1,
        neighbor=neighbor,
        steps=200,
        polish=40,
        device=device,
    )
    if float(result["W_upper"]) > previous + 1e-8:
        return f"kept {name} previous {previous:.6f} new {result['W_upper']:.6f}"
    payload = dict(
        model=model,
        n=n,
        boundary=boundary,
        params_json=json.dumps(params),
        beta=float(beta),
        alpha=float(alpha),
        seed=0,
        restarts=16,
        warm_start="alpha1_axes_repolish",
        W_upper=result["W_upper"],
        certified_global=result["certified_global"],
        axes_best=result["axes_best"],
        W_restarts=result["W_restarts"],
        axes_restarts=result["axes_restarts"],
        gradient_norm=result["gradient_norm"],
        gradient_within_tol=result["gradient_within_tol"],
        moment_disagreement=result["moment_disagreement"],
        W_alpha0=result["W_alpha0"],
        fixed_W=result["fixed_W"],
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
        device=str(device),
    )
    np.savez_compressed(destination, **payload)
    return (
        f"wrote {name} in {payload['runtime_seconds']:.1f}s "
        f"W_upper {result['W_upper']:.6f} was {previous:.6f} "
        f"grad {float(np.max(result['gradient_norm'])):.3e}"
    )


def main() -> None:
    devices = ["cuda:5", "cuda:6", "cuda:7"]
    tasks = [(*item, devices[index % 3]) for index, item in enumerate(TASKS)]
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        futures = [pool.submit(_one, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
