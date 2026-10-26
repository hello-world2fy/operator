"""Exact staggered-±Z Walshness for the classical Ising chain.

The n=12 multi-start search returns this texture. The transfer matrix
evaluates that one basis exactly. The number is the intrinsic minimum when
it equals the floor n, and otherwise only an upper bound.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.grid import ALPHAS, BETAS, TAIL_EPSILONS, case_name
from operator_walshness.integrable import (
    classical_chain_walshness,
    classical_staggered_z_spectrum,
    classical_staggered_z_walshness,
)
from operator_walshness.optimize import _staggered
from operator_walshness.walsh import identity_walshness, spectrum_moments, tail_degree


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_basis(out: Path, n: int, boundary: str, beta: float, alpha: float) -> None:
    name = case_name("commuting_zz_ising", n, boundary, {}) + f"_b{beta:g}_a{alpha:g}_s0"
    destination = out / f"{name}.npz"
    if destination.exists():
        print(f"skip {name}", flush=True)
        return
    value = classical_staggered_z_walshness(n, beta, alpha, boundary)
    fixed = classical_chain_walshness(n, beta, alpha, boundary)
    floor = float(n)
    certified = abs(value - floor) <= 1e-8 or abs(beta) == 0.0
    if abs(beta) == 0.0 and abs(value - identity_walshness(n, alpha)) > 1e-8:
        raise SystemExit(f"identity mismatch {name}")
    axes = _staggered(n, 2)
    np.savez_compressed(
        destination,
        model="commuting_zz_ising",
        n=n,
        boundary=boundary,
        params_json="{}",
        beta=float(beta),
        alpha=float(alpha),
        seed=0,
        restarts=1,
        W_upper=value,
        certified_global=certified,
        axes_best=axes,
        W_restarts=np.array([value]),
        axes_restarts=axes[None, :, :],
        gradient_norm=np.full(n, np.nan),
        gradient_within_tol=False,
        moment_disagreement=0.0,
        W_alpha0=floor,
        fixed_W=fixed,
        code_hash=_code_hash(),
        runtime_seconds=0.0,
        device="transfer_matrix",
        method="staggered_z_transfer",
    )
    print(f"wrote {name} W {value:.6f} fixed {fixed:.6f} certified {certified}", flush=True)


def _write_large(out: Path, n: int, boundary: str) -> None:
    name = case_name("commuting_zz_ising_staggered_z", n, boundary, {})
    destination = out / f"{name}.npz"
    if destination.exists():
        print(f"skip {name}", flush=True)
        return
    alphas = np.array(ALPHAS, dtype=np.float64)
    betas = np.array(BETAS, dtype=np.float64)
    walsh = np.zeros((betas.size, alphas.size))
    fixed = np.zeros_like(walsh)
    spectra = []
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            walsh[i, j] = classical_staggered_z_walshness(n, float(beta), float(alpha), boundary)
            fixed[i, j] = classical_chain_walshness(n, float(beta), float(alpha), boundary)
        spectrum = classical_staggered_z_spectrum(n, float(beta), boundary)
        moments = spectrum_moments(spectrum)
        if abs(moments["mean_degree"] - n) > 1e-6:
            raise SystemExit(f"mean degree {moments['mean_degree']} != {n} for {name} beta {beta}")
        spectra.append((spectrum, moments))
    moment_rows = [item[1] for item in spectra]
    spectra = np.stack([item[0] for item in spectra])
    tails = np.array([[tail_degree(row, eps) for eps in TAIL_EPSILONS] for row in spectra])
    np.savez_compressed(
        destination,
        model="commuting_zz_ising",
        basis="staggered_z",
        n=n,
        boundary=boundary,
        betas=betas,
        alphas=alphas,
        W_staggered=walsh,
        W_fixed_x=fixed,
        spectrum=spectra,
        tails=tails,
        degree_mean=np.array([item["mean_degree"] for item in moment_rows]),
        degree_variance=np.array([item["variance"] for item in moment_rows]),
        code_hash=_code_hash(),
        note="Exact in the staggered ±Z basis. Intrinsic only where W equals n.",
    )
    print(f"wrote {name}", flush=True)


def main() -> None:
    basis = Path("results/round1/basis")
    large = Path("results/round1/integrable")
    basis.mkdir(parents=True, exist_ok=True)
    large.mkdir(parents=True, exist_ok=True)
    for boundary in ("open", "periodic"):
        for beta in BETAS:
            for alpha in (0.25, 0.5, 2.0):
                _write_basis(basis, 12, boundary, float(beta), float(alpha))
        for n in (32, 64, 128, 256, 512):
            _write_large(large, n, boundary)


if __name__ == "__main__":
    main()
