"""Warm-started basis search at α other than 1.

Each start set includes the saved α=1 axes for the same state. The returned
value is an upper bound. α=0 is already certified by the magnetization formula
and is not repeated here.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from operator_walshness.exact import diagonalize
from operator_walshness.grid import BETAS, EXACT_CASES, case_name
from operator_walshness.optimize import optimize_walshness
from operator_walshness.walsh import gibbs_density
from run_basis_opt import _code_hash

ALPHAS = (0.25, 0.5, 2.0)


def _alpha1_axes(model: str, n: int, boundary: str, params: dict, beta: float, out_dir: str) -> np.ndarray | None:
    path = Path(out_dir) / f"{case_name(model, n, boundary, params)}_b{beta:g}_a1_s0.npz"
    if not path.exists():
        return None
    return np.load(path)["axes_best"]


def _one(task: tuple) -> str:
    model, n, boundary, params, beta, alpha, device, out_dir = task
    name = case_name(model, n, boundary, params) + f"_b{beta:g}_a{alpha:g}_s0"
    destination = Path(out_dir) / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    started = time.time()
    evals, evecs, _ = diagonalize(model, n, boundary, params)
    density, _ = gibbs_density(evals, evecs, beta)
    neighbor = _alpha1_axes(model, n, boundary, params, beta, out_dir)
    if neighbor is None:
        return f"skip {name} no alpha=1 axes"
    # n=12 Adam is the expensive part. Forty steps plus a short quasi-Newton
    # polish is enough once the alpha=1 axes are the warm start.
    steps, polish = (40, 20) if n >= 12 else (80, 15)
    result = optimize_walshness(
        density,
        alpha,
        restarts=8,
        seed=0,
        neighbor=neighbor,
        steps=steps,
        polish=polish,
        device=device,
    )
    payload = dict(
        model=model,
        n=n,
        boundary=boundary,
        params_json=json.dumps(params),
        beta=float(beta),
        alpha=float(alpha),
        seed=0,
        restarts=8,
        warm_start="alpha1_axes",
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    gap = float(result["fixed_W"] - result["W_upper"])
    return f"wrote {name} in {payload['runtime_seconds']:.1f}s W_upper {result['W_upper']:.6f} fixed {result['fixed_W']:.6f} gap {gap:.3e}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=[6, 8])
    parser.add_argument("--devices", nargs="*", default=["cuda:5", "cuda:6", "cuda:7"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--match", default=None, help="Substring of the case name. A trailing $ requires the name to end with the text.")
    args = parser.parse_args()
    devices = args.devices
    selected = None if args.models is None else set(args.models)
    out = str(Path(__file__).resolve().parents[1] / "results" / "round1" / "basis")
    tasks = []
    for model, boundary, params in EXACT_CASES:
        if selected is not None and model not in selected:
            continue
        for n in args.sizes:
            if model == "disconnected_dimers" and n % 2:
                continue
            for beta in BETAS:
                for alpha in ALPHAS:
                    name = case_name(model, n, boundary, dict(params))
                    needle = args.match
                    if needle:
                        if needle.endswith("$") and not name.endswith(needle[:-1]):
                            continue
                        if not needle.endswith("$") and needle not in name:
                            continue
                    tasks.append((model, n, boundary, dict(params), float(beta), float(alpha), devices[len(tasks) % len(devices)], out))
    print(f"tasks {len(tasks)}", flush=True)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(_one, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
