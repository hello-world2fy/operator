"""Exact Operator Walshness census. One file per Hamiltonian."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import diagonalize, eigenstate_table, gibbs_record, microcanonical_table
from operator_walshness.grid import ALPHAS, BETAS, EXACT_2D, EXACT_CASES, EXACT_SIZES, EXCITED_MODELS, TAIL_EPSILONS, case_name


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_one(task: tuple) -> str:
    model, n, boundary, params, out_dir, excited = task
    started = time.time()
    name = case_name(model, n, boundary, params)
    destination = Path(out_dir) / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    evals, evecs, sector = diagonalize(model, n, boundary, params)
    spectra = []
    walsh = []
    tails = []
    thermo_energy = []
    thermo_var = []
    thermo_cv = []
    purity = []
    r_eff = []
    magnetization = []
    alpha0 = []
    alpha0_axes = []
    product_error = []
    osee = []
    corr_x = []
    corr_z = []
    degree_mean = []
    degree_corr = []
    mean_degree = []
    variance = []
    mass_error = []
    schmidt_chunks = []
    for beta in BETAS:
        record = gibbs_record(evals, evecs, beta, ALPHAS)
        spectra.append(record["spectrum"])
        walsh.append(record["W_fixed"])
        tails.append(record["tails"])
        thermo_energy.append(record["thermo"]["energy"])
        thermo_var.append(record["thermo"]["energy_variance"])
        thermo_cv.append(record["thermo"]["heat_capacity"])
        purity.append(record["thermo"]["purity"])
        r_eff.append(record["thermo"]["r_eff"])
        magnetization.append(record["magnetization"])
        alpha0.append(record["W_alpha0_intrinsic"])
        alpha0_axes.append(record["alpha0_axes"])
        product_error.append(record["product_relative_error"])
        osee.append(record["osee"])
        corr_x.append(record["correlation_x"])
        corr_z.append(record["correlation_z"])
        degree_mean.append(record["degree_mean"])
        degree_corr.append(record["degree_correlation"])
        mean_degree.append(record["moments"]["mean_degree"])
        variance.append(record["moments"]["variance"])
        mass_error.append(record["diagnostics"]["mass_relative_error"])
        schmidt_chunks.append(record["schmidt"])
    payload = dict(
        model=model,
        n=n,
        boundary=boundary,
        params_json=json.dumps(params),
        betas=np.array(BETAS),
        alphas=np.array(ALPHAS),
        tail_epsilons=np.array(TAIL_EPSILONS),
        eigenvalues=evals,
        spectrum=np.stack(spectra),
        W_fixed=np.stack(walsh),
        tails=np.stack(tails),
        energy=np.array(thermo_energy),
        energy_variance=np.array(thermo_var),
        heat_capacity=np.array(thermo_cv),
        purity=np.array(purity),
        r_eff=np.array(r_eff),
        magnetization=np.stack(magnetization),
        W_alpha0_intrinsic=np.array(alpha0),
        alpha0_axes=np.stack(alpha0_axes),
        product_relative_error=np.stack(product_error),
        osee=np.array(osee),
        correlation_x=np.stack(corr_x),
        correlation_z=np.stack(corr_z),
        degree_mean=np.stack(degree_mean),
        degree_correlation=np.stack(degree_corr),
        mean_degree=np.array(mean_degree),
        degree_variance=np.array(variance),
        mass_relative_error=np.array(mass_error),
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    if excited and model in EXCITED_MODELS and boundary == "open":
        eigen = eigenstate_table(evals, evecs, sector, ALPHAS)
        micro = microcanonical_table(evals, evecs, ALPHAS)
        payload.update({f"eigen_{key}": value for key, value in eigen.items()})
        payload.update({f"micro_{key}": value for key, value in micro.items()})
    # Schmidt spectra have different ranks. Store a ragged object array.
    payload["schmidt"] = np.array(schmidt_chunks, dtype=object)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    max_error = float(np.max(payload["product_relative_error"]))
    return f"wrote {name} in {time.time() - started:.1f}s max_formula_rel {max_error:.3e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/round1/exact"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(EXACT_SIZES))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--excited", action="store_true")
    parser.add_argument("--include-2d", action="store_true")
    args = parser.parse_args()
    tasks = []
    for model, boundary, params in EXACT_CASES:
        if args.models and model not in args.models:
            continue
        for n in args.sizes:
            if model == "disconnected_dimers" and n % 2:
                continue
            tasks.append((model, n, boundary, dict(params), str(args.out), args.excited))
    if args.include_2d:
        for lx, ly in EXACT_2D:
            tasks.append(("ising_2d_classical", lx * ly, "open", {"lx": lx, "ly": ly}, str(args.out), False))
    metadata = {
        "code_hash": _code_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tasks": len(tasks),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.workers == 1:
        for task in tasks:
            print(_run_one(task), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
