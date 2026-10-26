"""Flatten the round-1 npz files into scalar tables.

Arrays stay in the npz files. These tables are the index for Walshness,
errors, and the thermal observables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "round1"


def _exact_rows() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "exact").glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        if "W_fixed" not in data.files:
            continue
        alphas = np.array(data["alphas"], dtype=float)
        betas = np.array(data["betas"], dtype=float)
        for beta_index, beta in enumerate(betas):
            for alpha_index, alpha in enumerate(alphas):
                rows.append(
                    {
                        "model": str(data["model"]),
                        "n": int(data["n"]),
                        "boundary": str(data["boundary"]),
                        "params_json": str(data["params_json"]),
                        "beta": float(beta),
                        "alpha": float(alpha),
                        "W_fixed": float(data["W_fixed"][beta_index, alpha_index]),
                        "density": float(data["W_fixed"][beta_index, alpha_index]) / (2 * int(data["n"])),
                        "energy": float(data["energy"][beta_index]),
                        "purity": float(data["purity"][beta_index]),
                        "osee_bits": float(data["osee"][beta_index]),
                        "mean_degree": float(data["mean_degree"][beta_index]),
                        "W_alpha0_intrinsic": float(data["W_alpha0_intrinsic"][beta_index]),
                        "mass_relative_error": float(data["mass_relative_error"][beta_index]),
                    }
                )
    return rows


def _basis_rows() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "basis").glob("*.npz")):
        data = np.load(path)
        restarts = np.array(data["W_restarts"], dtype=float)
        rows.append(
            {
                "model": str(data["model"]),
                "n": int(data["n"]),
                "boundary": str(data["boundary"]),
                "params_json": str(data["params_json"]),
                "beta": float(data["beta"]),
                "alpha": float(data["alpha"]),
                "seed": int(data["seed"]),
                "W_upper": float(data["W_upper"]),
                "fixed_W": float(data["fixed_W"]),
                "gap": float(data["fixed_W"]) - float(data["W_upper"]),
                "certified_global": bool(data["certified_global"]),
                "restart_spread": float(np.ptp(restarts)),
                "moment_disagreement": float(data["moment_disagreement"]),
                "gradient_within_tol": bool(data["gradient_within_tol"]),
            }
        )
    return rows


def _mpo_rows() -> list[dict]:
    rows = []
    folder = ROOT / "mpo"
    if not folder.exists():
        return rows
    for path in sorted(folder.glob("*.npz")):
        data = np.load(path)
        alphas = np.array(data["alphas"], dtype=float)
        errors = np.array(data["W_abs_error"], dtype=float)
        for index, alpha in enumerate(alphas):
            rows.append(
                {
                    "model": str(data["model"]),
                    "n": int(data["n"]),
                    "beta": float(data["beta"]),
                    "alpha": float(alpha),
                    "chi_max": int(data["chi_max"]),
                    "dt": float(data["dt_requested"]),
                    "W_log": float(data["W_log"][index]),
                    "density": float(data["W_log"][index]) / (2 * int(data["n"])),
                    "discarded_weight": float(data["discarded_weight"]),
                    "osee_bits": float(data["osee"]),
                    "reference_method": str(data["reference_method"]),
                    "W_abs_error": float(errors[index]),
                    "energy": float(data["energy"]),
                }
            )
    return rows


def main() -> None:
    out = ROOT / "scalars"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "exact_gibbs": _exact_rows(),
        "basis_upper": _basis_rows(),
        "mpo": _mpo_rows(),
    }
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        frame.to_parquet(out / f"{name}.parquet", index=False)
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"{name} {len(frame)} rows", flush=True)
    (out / "README.json").write_text(
        json.dumps(
            {
                "note": "W_upper is an upper bound except where certified_global is true. MPO W_abs_error is missing when reference_method is none. discarded_weight is the sum of per-SVD discarded fractions and can exceed 1.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
