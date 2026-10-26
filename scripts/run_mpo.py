"""Fixed-basis Operator Walshness from an imaginary-time MPO.

One file per model, size, bond dimension, Trotter step and temperature.
W_α at α>0 is the log-contraction, not the Fourier tail. The spectrum, OSEE,
discarded weight and a reference error are stored beside it. A small χ
difference is a screening check, not a proof that the Trotter error is small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.free_fermion import OpenBoundaryTFIM
from operator_walshness.grid import ALPHAS, BETAS, TAIL_EPSILONS, case_name
from operator_walshness.integrable import classical_chain_walshness
from operator_walshness.mpo import (
    degree_spectrum,
    evolve_mpo,
    fixed_walshness,
    operator_entropy,
    thermal_observables,
)
from operator_walshness.walsh import spectrum_moments, tail_degree, walshness_from_spectrum

CASES = [
    ("transverse_field_ising", {"g": 1.0}),
    ("mixed_field_ising", {"g": 0.8, "h": 0.2}),
    ("mixed_field_ising", {"g": 1.05, "h": 0.2}),
    ("mixed_field_ising", {"g": 1.05, "h": 0.5}),
    ("mixed_field_ising", {"g": 1.2, "h": 0.5}),
    ("xxz", {"delta": 0.5}),
    ("xxz", {"delta": 1.0}),
    ("commuting_zz_ising", {}),
]


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _beta_tag(beta: float) -> str:
    return f"{float(beta):.6g}"


def _archived_reference(model: str, n: int, params: dict, beta: float) -> dict | None:
    """Use a saved free-fermion or transfer-matrix file, including its spectrum."""
    archive = Path(__file__).resolve().parents[1] / "results" / "round1" / "integrable"
    path = archive / f"{case_name(model, n, 'open', params)}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    matches = np.where(np.isclose(data["betas"], beta))[0]
    if matches.size == 0:
        return None
    index = int(matches[0])
    spectrum = None
    if "spectrum" in data.files and bool(np.array(data["spectrum_computed"])[index]):
        spectrum = np.array(data["spectrum"][index], dtype=float)
    energy = float(data["energy"][index]) if "energy" in data.files else np.nan
    return {
        "method": str(data["method"]),
        "W": np.array(data["W_fixed"][index], dtype=float),
        "spectrum": spectrum,
        "osee": np.nan,
        "energy": energy,
        "log_purity": np.nan,
    }


def _reference(model: str, n: int, params: dict, beta: float) -> dict:
    exact_path = Path(__file__).resolve().parents[1] / "results" / "round1" / "exact"
    exact_file = exact_path / f"{case_name(model, n, 'open', params)}.npz"
    alphas = np.array(ALPHAS, dtype=float)
    if exact_file.exists():
        data = np.load(exact_file)
        index = int(np.where(np.isclose(data["betas"], beta))[0][0])
        return {
            "method": "exact",
            "W": np.array(data["W_fixed"][index], dtype=float),
            "spectrum": np.array(data["spectrum"][index], dtype=float),
            "osee": float(data["osee"][index]),
            "energy": float(data["energy"][index]),
            "log_purity": float(np.log(data["purity"][index])),
        }
    archived = _archived_reference(model, n, params, beta)
    if archived is not None:
        return archived
    if model == "transverse_field_ising":
        fermion = OpenBoundaryTFIM(n, float(params["g"]))
        beta = float(beta)
        thermal = -0.5 * fermion.energies * np.tanh(0.5 * beta * fermion.energies)
        spectrum = fermion.degree_spectrum(beta) if n <= 32 else None
        return {
            "method": "free_fermion",
            "W": np.array([fermion.fixed_walshness(beta, alpha) for alpha in alphas]),
            "spectrum": spectrum,
            "osee": np.nan,
            "energy": float(np.sum(thermal)),
            "log_purity": float(fermion.log_partition(2.0 * beta) - 2.0 * fermion.log_partition(beta)),
        }
    if model == "commuting_zz_ising":
        return {
            "method": "transfer_matrix",
            "W": np.array([classical_chain_walshness(n, beta, alpha, "open") for alpha in alphas]),
            "spectrum": None,
            "osee": np.nan,
            "energy": np.nan,
            "log_purity": np.nan,
        }
    return {
        "method": "none",
        "W": np.full(alphas.shape, np.nan),
        "spectrum": None,
        "osee": np.nan,
        "energy": np.nan,
        "log_purity": np.nan,
    }


def _dispatch(task: tuple) -> str:
    try:
        return _dispatch_one(task)
    except Exception as exc:
        model, params, n, chi, dt, beta, _out = task
        tag = f"{case_name(model, n, 'open', params)}_chi{chi}_dt{_beta_tag(dt)}_b{_beta_tag(beta)}"
        return f"FAILED {tag}: {exc}"


def _dispatch_one(task: tuple) -> str:
    model, params, n, chi, dt, beta, out = task
    started = time.time()
    name = case_name(model, n, "open", params)
    tag = f"{name}_chi{chi}_dt{_beta_tag(dt)}_b{_beta_tag(beta)}"
    destination = Path(out) / f"{tag}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {tag}"
    g = float(params.get("g", 1.0))
    h = float(params.get("h", 0.0))
    delta = float(params.get("delta", 1.0))
    tensors, stats = evolve_mpo(model, n, beta, dt, g=g, h=h, delta=delta, chi_max=chi, cutoff=1e-12)
    alphas = np.array(ALPHAS, dtype=float)
    spectrum = degree_spectrum(tensors)
    walsh_log = np.array([fixed_walshness(tensors, alpha) for alpha in alphas])
    if not np.all(np.isfinite(walsh_log)):
        raise RuntimeError(f"non-finite Walshness at beta={beta}")
    walsh_spectrum = np.array([walshness_from_spectrum(spectrum, alpha) for alpha in alphas])
    moments = spectrum_moments(spectrum)
    tails = np.array([tail_degree(spectrum, eps) for eps in TAIL_EPSILONS])
    weights, entropy = operator_entropy(tensors)
    thermal = thermal_observables(tensors, model, g, h, delta)
    reference = _reference(model, n, params, beta)
    walsh_error = np.abs(walsh_log - reference["W"])
    if reference["spectrum"] is None:
        spectrum_l1 = np.nan
    else:
        spectrum_l1 = float(np.abs(spectrum - reference["spectrum"]).sum())
    payload = dict(
        model=model,
        n=n,
        boundary="open",
        params_json=json.dumps(params),
        method="mpo_trotter",
        beta=float(beta),
        alphas=alphas,
        tail_epsilons=np.array(TAIL_EPSILONS),
        chi_max=int(chi),
        dt_requested=float(dt),
        dt_used=float(stats["dt"]),
        steps=int(stats["steps"]),
        largest_chi=int(stats["largest_chi"]),
        chi_profile=np.array(stats.get("chi_profile", []), dtype=np.int32),
        discarded_weight=float(stats["discarded_weight"]),
        max_discarded_weight=float(stats.get("max_discarded_weight", 0.0)),
        spectrum=spectrum,
        W_log=walsh_log,
        W_spectrum=walsh_spectrum,
        spectrum_moment_gap=float(abs(walsh_spectrum[0] - walsh_log[0])),
        tails=tails,
        mean_degree=float(moments["mean_degree"]),
        degree_variance=float(moments["variance"]),
        osee=float(entropy),
        schmidt=weights[weights > 1e-12],
        energy=float(thermal["energy"]),
        log_purity=float(thermal["log_purity"]),
        magnetization=thermal["magnetization"],
        reference_method=reference["method"],
        reference_W=reference["W"],
        W_abs_error=walsh_error,
        spectrum_l1=spectrum_l1,
        osee_abs_error=float(abs(entropy - reference["osee"])) if np.isfinite(reference["osee"]) else np.nan,
        energy_abs_error=float(abs(thermal["energy"] - reference["energy"])) if np.isfinite(reference["energy"]) else np.nan,
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    worst = float(np.nanmax(walsh_error))
    return (
        f"wrote {tag} in {payload['runtime_seconds']:.1f}s "
        f"chi {payload['largest_chi']} disc {payload['discarded_weight']:.2e} "
        f"W1 {walsh_log[3]:.6f} ref {reference['method']} err {worst:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/round1/mpo"))
    parser.add_argument("--sizes", type=int, nargs="*", default=[12, 32, 64])
    parser.add_argument("--chi", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--dt", type=float, nargs="*", default=[0.05, 0.1])
    parser.add_argument("--chi-sweep-max-n", type=int, default=32)
    parser.add_argument("--dt-sweep-max-n", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--betas", type=float, nargs="*", default=list(BETAS))
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    cases = [item for item in CASES if args.models is None or item[0] in set(args.models)]
    tasks = []
    for n in args.sizes:
        chis = args.chi if n <= args.chi_sweep_max_n else args.chi[:1]
        dts = args.dt if n <= args.dt_sweep_max_n else args.dt[:1]
        for model, params in cases:
            for chi in chis:
                for dt in dts:
                    for beta in args.betas:
                        tasks.append((model, params, n, chi, dt, float(beta), str(args.out)))
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "code_hash": _code_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tasks": len(tasks),
        "note": "W_log is the reported fixed-basis Walshness. chi and dt comparisons are screening checks.",
    }
    meta_path = args.out / "metadata.json"
    if meta_path.exists():
        tag = "-".join(str(size) for size in args.sizes)
        meta_path = args.out / f"metadata_n{tag}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"tasks {len(tasks)}", flush=True)
    if args.workers == 1:
        for task in tasks:
            print(_dispatch(task), flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_dispatch, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
