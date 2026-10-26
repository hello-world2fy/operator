"""Fill product-spin and dimer basis files from exact reductions.

Product spins are separable. The fixed-X value is the unique minimum for
g>0, so certified_global is true. Dimers factor across disjoint pairs, so
the n-site upper bound is (n/2) times a fully optimized two-site upper bound.
That dimer value is still an upper bound.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize
from operator_walshness.grid import BETAS, case_name
from operator_walshness.optimize import optimize_walshness
from operator_walshness.walsh import fixed_x_directions, gibbs_density


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def product_walshness(n: int, g: float, beta: float, alpha: float) -> float:
    scale = 2.0 * float(g) * float(beta)
    if float(alpha) == 0.0:
        return float(n * (1.0 - np.tanh(scale)))
    log_moment = np.logaddexp(scale, -scale + 2.0 * float(alpha)) - np.logaddexp(scale, -scale)
    return float(n * log_moment / float(alpha))


def _check_product(out: Path, n: int, g: float, beta: float, alpha: float) -> None:
    path = out / f"{case_name('one_site_product', n, 'open', {'g': g})}_b{beta:g}_a{alpha:g}_s0.npz"
    if not path.exists():
        return
    stored = float(np.load(path)["W_upper"])
    exact = product_walshness(n, g, beta, alpha)
    if abs(stored - exact) > 1e-6:
        raise SystemExit(f"product check failed {path.name}: stored {stored} exact {exact}")


def _write_product(out: Path, n: int, g: float, beta: float, alpha: float) -> None:
    name = case_name("one_site_product", n, "open", {"g": g}) + f"_b{beta:g}_a{alpha:g}_s0"
    destination = out / f"{name}.npz"
    if destination.exists():
        print(f"skip {name}", flush=True)
        return
    value = product_walshness(n, g, beta, alpha)
    alpha0 = product_walshness(n, g, beta, 0.0)
    axes = fixed_x_directions(n)
    np.savez_compressed(
        destination,
        model="one_site_product",
        n=n,
        boundary="open",
        params_json=json.dumps({"g": float(g)}),
        beta=float(beta),
        alpha=float(alpha),
        seed=0,
        restarts=1,
        W_upper=value,
        certified_global=True,
        axes_best=axes,
        W_restarts=np.array([value]),
        axes_restarts=axes[None, :, :],
        gradient_norm=np.zeros(n),
        gradient_within_tol=True,
        moment_disagreement=0.0,
        W_alpha0=alpha0,
        fixed_W=value,
        code_hash=_code_hash(),
        runtime_seconds=0.0,
        device="closed_form",
        method="product_closed_form",
    )
    print(f"wrote {name} W {value:.6f} certified", flush=True)


def _dimer_two_site(beta: float, alpha: float) -> dict:
    evals, evecs, _ = diagonalize("disconnected_dimers", 2, "open", {})
    density, _ = gibbs_density(evals, evecs, beta)
    return optimize_walshness(
        density,
        alpha,
        restarts=24,
        seed=0,
        steps=200,
        polish=30,
        device="cpu",
    )


def _check_dimer(out: Path, beta: float, alpha: float, two: dict) -> None:
    path = out / f"{case_name('disconnected_dimers', 8, 'open', {})}_b{beta:g}_a{alpha:g}_s0.npz"
    if not path.exists():
        return
    stored = float(np.load(path)["W_upper"])
    scaled = 4.0 * float(two["W_upper"])
    # The eight-site search is an independent upper bound. It must not sit
    # below the factored two-site bound by more than optimizer noise.
    if stored + 1e-5 < scaled:
        raise SystemExit(f"dimer n=8 {stored} is below 4 * n=2 {scaled} at beta {beta}")
    print(f"dimer check beta {beta:g}: n8 {stored:.6f} 4*n2 {scaled:.6f}", flush=True)


def _write_dimer(out: Path, n: int, beta: float, alpha: float, two: dict) -> None:
    copies = n // 2
    name = case_name("disconnected_dimers", n, "open", {}) + f"_b{beta:g}_a{alpha:g}_s0"
    destination = out / f"{name}.npz"
    if destination.exists():
        print(f"skip {name}", flush=True)
        return
    axes = np.tile(two["axes_best"], (copies, 1))
    restarts = np.tile(two["axes_restarts"], (1, copies, 1))
    np.savez_compressed(
        out / f"{name}.npz",
        model="disconnected_dimers",
        n=n,
        boundary="open",
        params_json="{}",
        beta=float(beta),
        alpha=float(alpha),
        seed=0,
        restarts=int(two["W_restarts"].shape[0]),
        W_upper=copies * float(two["W_upper"]),
        certified_global=False,
        axes_best=axes,
        W_restarts=copies * np.asarray(two["W_restarts"]),
        axes_restarts=restarts,
        gradient_norm=np.tile(two["gradient_norm"], copies),
        gradient_within_tol=bool(two["gradient_within_tol"]),
        moment_disagreement=float(two["moment_disagreement"]),
        W_alpha0=copies * float(two["W_alpha0"]),
        fixed_W=copies * float(two["fixed_W"]),
        code_hash=_code_hash(),
        runtime_seconds=0.0,
        device="n2_additivity",
        method="dimer_additivity_from_n2",
    )
    print(f"wrote {name} W_upper {copies * float(two['W_upper']):.6f}", flush=True)


def main() -> None:
    out = Path("results/round1/basis")
    out.mkdir(parents=True, exist_ok=True)
    for alpha in (0.25, 0.5, 1.0, 2.0):
        for g in (0.5, 1.0):
            for beta in BETAS:
                _check_product(out, 8, g, beta, alpha)
                _write_product(out, 12, g, beta, alpha)
        for beta in BETAS:
            two = _dimer_two_site(float(beta), alpha)
            _check_dimer(out, float(beta), alpha, two)
            _write_dimer(out, 12, float(beta), alpha, two)


if __name__ == "__main__":
    main()
