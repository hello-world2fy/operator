"""Continue a saved n=12 basis search from its stored restarts.

The first pass uses 40 Adam steps and no LBFGS, so several files miss the
gradient tolerance. This script takes more Adam steps from those axes and
replaces a file only when the NumPy Walshness decreases. The result is still
an upper bound unless it lands on the alpha->0 floor.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize
from operator_walshness.optimize import _adam_batch, _lbfgs_one, riemannian_gradient
from operator_walshness.walsh import fixed_x_directions, gibbs_density, walshness_from_directions


def _polish_one(path: Path, device: str, steps: int, top: int) -> str:
    data = np.load(path)
    if bool(data["gradient_within_tol"]):
        return f"skip {path.name}"
    params = json.loads(str(data["params_json"]))
    alpha = float(data["alpha"])
    evals, evecs, _ = diagonalize(str(data["model"]), int(data["n"]), str(data["boundary"]), params)
    operator, _ = gibbs_density(evals, evecs, float(data["beta"]))
    torch_device = torch.device(device)
    op = torch.tensor(operator, dtype=torch.complex128, device=torch_device)
    starts = np.array(data["axes_restarts"])
    order = np.argsort(np.array(data["W_restarts"]))
    if top > 0:
        starts = starts[order[: min(top, starts.shape[0])]]
    started = time.time()
    if steps < 0:
        refined = np.stack([_lbfgs_one(op, axes, alpha, torch_device, -steps) for axes in starts], axis=0)
    else:
        refined = _adam_batch(op, starts, alpha, steps, 0.05, torch_device)
    values = np.array([walshness_from_directions(operator, axes, alpha) for axes in refined])
    best = int(np.argmin(values))
    chosen = refined[best]
    chosen_value = float(values[best])
    fixed_value = float(data["fixed_W"])
    # The all-+X basis is always feasible. Keep it when the search walks above it.
    if fixed_value + 1e-12 < chosen_value:
        chosen = fixed_x_directions(int(data["n"]))
        chosen_value = fixed_value
    previous = float(data["W_upper"])
    if chosen_value >= previous - 1e-12:
        return f"keep {path.name} W {previous:.6f} new {chosen_value:.6f}"
    gradients = riemannian_gradient(op, chosen, alpha, torch_device)
    n = int(data["n"])
    kept_axes = np.array(data["axes_restarts"])
    kept_values = np.array(data["W_restarts"], dtype=float)
    # Retain starts that were not continued, so the restart distribution stays intact.
    if top > 0 and kept_values.shape[0] > starts.shape[0]:
        kept_order = np.argsort(kept_values)
        kept_axes = np.concatenate([refined, kept_axes[kept_order[starts.shape[0] :]]], axis=0)
        kept_values = np.concatenate([values, kept_values[kept_order[starts.shape[0] :]]])
    else:
        kept_axes = refined
        kept_values = values
    payload = {key: data[key] for key in data.files}
    payload["W_upper"] = chosen_value
    payload["axes_best"] = chosen
    payload["W_restarts"] = kept_values
    payload["axes_restarts"] = kept_axes
    payload["gradient_norm"] = np.linalg.norm(gradients, axis=-1)
    payload["gradient_within_tol"] = bool(np.linalg.norm(gradients) <= 1e-7 * max(n, 1) ** 0.5)
    payload["certified_global"] = bool(abs(chosen_value - float(data["W_alpha0"])) <= 1e-8)
    payload["runtime_seconds"] = float(data["runtime_seconds"]) + time.time() - started
    np.savez_compressed(path, **payload)
    return (
        f"wrote {path.name} {previous:.6f} -> {chosen_value:.6f} "
        f"|grad| {float(np.linalg.norm(gradients)):.3e} in {time.time() - started:.1f}s"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--steps", type=int, default=80, help="Adam steps. A negative value runs that many LBFGS iterations instead.")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for path in args.files:
        print(_polish_one(path, args.device, args.steps, args.top), flush=True)


if __name__ == "__main__":
    main()
