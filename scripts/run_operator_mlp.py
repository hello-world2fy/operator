"""Small operator-MLP pilot.

The network fits the matrix elements of a Gibbs operator, optionally after a
local basis change. Every run uses the same number of Adam steps. The reported
error is the relative Frobenius residual on the whole matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize
from operator_walshness.grid import case_name
from operator_walshness.walsh import (
    align_direction_to_z,
    fixed_basis_spectrum,
    gibbs_density,
    rotate_operator,
    tail_degree,
)

TAIL_EPSILONS = (1e-2, 1e-3, 1e-4)


def _features(n: int) -> np.ndarray:
    index = np.arange(1 << n)
    bits = ((index[:, None] >> np.arange(n)) & 1).astype(np.float64)
    spins = 1.0 - 2.0 * bits
    ket = np.repeat(spins, 1 << n, axis=0)
    bra = np.tile(spins, (1 << n, 1))
    return np.concatenate([ket, bra], axis=1)


def _directions(kind: str, n: int, model: str, boundary: str, params: dict, beta: float, alpha: float) -> np.ndarray | None:
    if kind == "fixed_z":
        return None
    if kind == "random":
        rng = np.random.default_rng(12345)
        draw = rng.normal(size=(n, 3))
        return draw / np.linalg.norm(draw, axis=1, keepdims=True)
    path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "round1"
        / "basis"
        / f"{case_name(model, n, boundary, params)}_b{beta:g}_a{alpha:g}_s0.npz"
    )
    if not path.exists():
        return None
    return np.load(path)["axes_best"]


def _target(operator: np.ndarray, directions: np.ndarray | None) -> np.ndarray:
    if directions is None:
        matrix = np.asarray(operator, dtype=np.complex128)
    else:
        unitaries = [align_direction_to_z(directions[site]) for site in range(directions.shape[0])]
        matrix = rotate_operator(operator, unitaries)
    return matrix.reshape(-1)


class OperatorMLP(nn.Module):
    def __init__(self, n_inputs: int, width: int, depth: int, activation: str):
        super().__init__()
        self.activation_name = activation
        if activation == "square":
            if depth != 1:
                raise ValueError("the square network is one hidden layer")
            self.hidden = nn.Linear(n_inputs, width)
            self.readout = nn.Linear(width, 2)
        elif activation == "tanh":
            layers = []
            dim = n_inputs
            for _ in range(depth):
                layers.append(nn.Linear(dim, width))
                dim = width
            self.hidden_layers = nn.ModuleList(layers)
            self.readout = nn.Linear(dim, 2)
        else:
            raise ValueError(activation)
        for parameter in self.parameters():
            parameter.data.mul_(1e-2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "square":
            hidden = self.hidden(features).square()
        else:
            hidden = features
            for layer in self.hidden_layers:
                hidden = torch.tanh(layer(hidden))
        return self.readout(hidden)


def _fit(features: np.ndarray, target: np.ndarray, width: int, depth: int, activation: str, seed: int, steps: int, device: str) -> float:
    torch.manual_seed(seed)
    model = OperatorMLP(features.shape[1], width, depth, activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(np.stack([target.real, target.imag], axis=1), dtype=torch.float32, device=device)
    y = y / torch.sqrt(torch.sum(y * y)).clamp_min(1e-30)
    norm = torch.sum(y * y).clamp_min(1e-30)
    for _ in range(steps):
        optimizer.zero_grad()
        residual = model(x) - y
        loss = torch.sum(residual * residual) / norm
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        residual = model(x) - y
        return float((torch.sum(residual * residual) / norm).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="*", default=[8])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("results/round1/mlp"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = (
        ("transverse_field_ising", "open", {"g": 1.0}),
        ("mixed_field_ising", "open", {"g": 0.8, "h": 0.2}),
    )
    betas = (0.25, 1.0, 4.0)
    bases = (
        ("fixed_z", 1.0),
        ("optimized", 0.5),
        ("optimized", 1.0),
        ("optimized", 2.0),
        ("random", 1.0),
    )
    architectures = (
        ("square", 1, 32),
        ("square", 1, 64),
        ("tanh", 1, 32),
        ("tanh", 2, 32),
        ("tanh", 3, 32),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    cache: dict[tuple, np.ndarray] = {}
    feature_cache: dict[int, np.ndarray] = {}
    for model, boundary, params in cases:
        for n in args.sizes:
            evals, evecs, _ = diagonalize(model, n, boundary, params)
            features = feature_cache.setdefault(n, _features(n))
            for beta in betas:
                key = (model, n, beta)
                if key not in cache:
                    density, _ = gibbs_density(evals, evecs, beta)
                    cache[key] = density
                density = cache[key]
                for basis, alpha in bases:
                    directions = _directions(basis, n, model, boundary, params, beta, alpha)
                    if basis == "optimized" and directions is None:
                        print(f"skip missing basis {model} n{n} b{beta:g} a{alpha:g}", flush=True)
                        continue
                    target = _target(density, directions)
                    spectrum, _ = fixed_basis_spectrum(target.reshape(1 << n, 1 << n))
                    tails = [tail_degree(spectrum, eps) for eps in TAIL_EPSILONS]
                    label = basis if basis != "optimized" else f"optimized_a{alpha:g}"
                    for activation, depth, width in architectures:
                        for seed in range(args.seeds):
                            started = time.time()
                            error = _fit(features, target, width, depth, activation, seed, args.steps, args.device)
                            row = {
                                "model": model,
                                "n": n,
                                "boundary": boundary,
                                "params_json": json.dumps(params),
                                "beta": beta,
                                "basis": label,
                                "activation": activation,
                                "depth": depth,
                                "width": width,
                                "seed": seed,
                                "steps": args.steps,
                                "relative_frobenius": error,
                                "tail_1e-2": tails[0],
                                "tail_1e-3": tails[1],
                                "tail_1e-4": tails[2],
                                "runtime_seconds": time.time() - started,
                            }
                            rows.append(row)
                            print(
                                f"{model} b{beta:g} {label} {activation} d{depth} w{width} s{seed} "
                                f"err {error:.3e} tail {tails[1]} in {row['runtime_seconds']:.1f}s",
                                flush=True,
                            )
    import pandas as pd

    frame = pd.DataFrame(rows)
    tag = "n" + "-".join(str(size) for size in args.sizes)
    frame.to_parquet(args.out / f"pilot_{tag}.parquet", index=False)
    frame.to_csv(args.out / f"pilot_{tag}.csv", index=False)
    print(f"rows {len(frame)}", flush=True)


if __name__ == "__main__":
    main()
