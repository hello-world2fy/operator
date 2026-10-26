"""Multi-start Bloch optimization of intrinsic Operator Walshness.

Every finite-α result is an upper bound: the run stores the full restart
distribution, and certified_global is true only for the α→0 closed form.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize
from operator_walshness.grid import BETAS, EXACT_CASES, case_name
from operator_walshness.optimize import optimize_walshness
from operator_walshness.walsh import gibbs_density


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _transferred_axes(model: str, n: int, boundary: str, params: dict, beta: float, alpha: float, seed: int, out_dir: str) -> np.ndarray | None:
    """Extend the best n-2 texture by copying its edge directions."""
    if n < 8:
        return None
    source = Path(out_dir) / f"{case_name(model, n - 2, boundary, params)}_b{beta:g}_a{alpha:g}_s{seed}.npz"
    if not source.exists():
        return None
    axes = np.load(source)["axes_best"]
    out = np.zeros((n, axes.shape[1]), dtype=np.float64)
    start = (n - axes.shape[0]) // 2
    out[start : start + axes.shape[0]] = axes
    if start:
        out[:start] = axes[0]
        out[start + axes.shape[0] :] = axes[-1]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-15)


def _one(task: tuple) -> str:
    try:
        return _one_point(task)
    except Exception as exc:
        model, n, boundary, params, beta, alpha, seed = task[:7]
        name = case_name(model, n, boundary, params) + f"_b{beta:g}_a{alpha:g}_s{seed}"
        return f"FAILED {name}: {exc}"


def _one_point(task: tuple) -> str:
    model, n, boundary, params, beta, alpha, seed, restarts, device, out_dir = task
    name = case_name(model, n, boundary, params) + f"_b{beta:g}_a{alpha:g}_s{seed}"
    destination = Path(out_dir) / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    started = time.time()
    print(f"start {name}", flush=True)
    evals, evecs, _ = diagonalize(model, n, boundary, params)
    density, _ = gibbs_density(evals, evecs, beta)
    neighbor = _transferred_axes(model, n, boundary, params, beta, alpha, seed, out_dir)
    # n=12 Adam leaves gradients near 1e-2. Twenty LBFGS iterations on the
    # best restart reach the gradient tolerance in about ten seconds and move
    # W by about 1e-3. The value remains an upper bound.
    steps, polish = (40, 20) if n >= 12 else (200, 30)
    result = optimize_walshness(
        density,
        alpha,
        restarts=restarts,
        seed=seed,
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
        beta=beta,
        alpha=alpha,
        seed=seed,
        restarts=restarts,
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
    return (
        f"wrote {name} in {time.time() - started:.1f}s "
        f"W_upper {result['W_upper']:.6f} fixed {result['fixed_W']:.6f} gap {gap:.3e} "
        f"spread {float(np.ptp(result['W_restarts'])):.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/round1/basis"))
    parser.add_argument("--sizes", type=int, nargs="*", default=[6, 8])
    parser.add_argument("--alphas", type=float, nargs="*", default=[1.0])
    parser.add_argument("--betas", type=float, nargs="*", default=list(BETAS))
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", nargs="*", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    tasks = []
    slot = 0
    selected = None if args.models is None else set(args.models)
    for model, boundary, params in EXACT_CASES:
        if selected is not None and model not in selected:
            continue
        for n in args.sizes:
            if model == "disconnected_dimers" and n % 2:
                continue
            for beta in args.betas:
                for alpha in args.alphas:
                    device = args.devices[slot % len(args.devices)]
                    slot += 1
                    tasks.append(
                        (model, n, boundary, dict(params), float(beta), float(alpha), args.seed, args.restarts, device, str(args.out))
                    )
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.json"
    if meta_path.exists():
        tag = "-".join(str(size) for size in args.sizes)
        if args.models:
            tag += "-" + "-".join(args.models)
        meta_path = args.out / f"metadata_{tag}.json"
    meta_path.write_text(
        json.dumps(
            {
                "code_hash": _code_hash(),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "tasks": len(tasks),
                "note": "W_upper is a numerical upper bound except where certified_global is true",
            },
            indent=2,
        )
    )
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(_one, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
