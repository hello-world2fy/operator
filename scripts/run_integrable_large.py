"""Large-n fixed-basis Operator Walshness for integrable and product models.

Transverse-field Ising uses the open-boundary free-fermion trace. Classical ZZ
uses the two-replica transfer matrix. Product fields and dimers factor exactly.
Walshness at α>0 is taken from log G. Degree distributions are stored as well;
their far tails are limited by Fourier roundoff, so the saved mean-degree error
is the acceptance check for a spectrum.
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

from operator_walshness.exact import diagonalize
from operator_walshness.free_fermion import OpenBoundaryTFIM
from operator_walshness.grid import ALPHAS, BETAS, TAIL_EPSILONS, case_name
from operator_walshness.integrable import (
    _classical_chain_log_generating_value,
    classical_chain_spectrum,
    convolve_copies,
)
from operator_walshness.walsh import (
    gibbs_density,
    identity_walshness,
    intrinsic_alpha0,
    spectrum_moments,
    tail_degree,
    walshness_from_spectrum,
)


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _centered_log_derivative(log_plus: float, log_minus: float, step: float) -> float:
    return float((log_plus - log_minus) / (2.0 * step))


def _pack_moments(spectrum: np.ndarray | None, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if spectrum is None:
        empty = np.full(alphas.shape, np.nan)
        return empty, np.full((len(TAIL_EPSILONS),), np.nan), {"mean_degree": np.nan, "variance": np.nan}
    walsh = np.array([walshness_from_spectrum(spectrum, alpha) for alpha in alphas])
    tails = np.array([tail_degree(spectrum, eps) for eps in TAIL_EPSILONS])
    moments = spectrum_moments(spectrum)
    return walsh, tails, moments


def _tfim_job(n: int, g: float, out_dir: Path) -> str:
    started = time.time()
    name = case_name("transverse_field_ising", n, "open", {"g": g})
    destination = out_dir / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    model = OpenBoundaryTFIM(n, g)
    spectra = []
    mask = []
    walsh_trace = []
    alpha0 = []
    magnetization = []
    energy = []
    variance = []
    heat = []
    log_purity = []
    mean_error = []
    for beta in BETAS:
        want_spectrum = n <= 128 or (g == 1.0 and beta in {0.0, 0.5, 1.0, 2.0, 4.0})
        row = np.zeros(2 * n + 1)
        if want_spectrum and n <= 256:
            row = model.degree_spectrum(beta)
            mask.append(True)
            mean_error.append(abs(float(np.dot(np.arange(row.size), row)) - model.fixed_walshness(beta, 0.0)))
        else:
            mask.append(False)
            mean_error.append(np.nan)
        spectra.append(row)
        walsh_trace.append([model.fixed_walshness(beta, alpha) for alpha in ALPHAS])
        magnet = model.magnetization(2.0 * beta)
        magnetization.append(model.magnetization(beta))
        alpha0.append(float(np.sum(1.0 - np.abs(magnet[:, 0]))))
        thermal = -0.5 * model.energies * np.tanh(0.5 * beta * model.energies)
        energy.append(float(np.sum(thermal)))
        var = float(np.sum((0.5 * model.energies / np.cosh(0.5 * beta * model.energies)) ** 2))
        variance.append(var)
        heat.append(float(beta**2 * var))
        log_purity.append(model.log_partition(2.0 * beta) - 2.0 * model.log_partition(beta))
    spectrum = np.stack(spectra)
    walsh_from_spectrum = []
    tails = []
    means = []
    variances = []
    for row, kept in zip(spectrum, mask):
        if not kept:
            walsh_from_spectrum.append(np.full(len(ALPHAS), np.nan))
            tails.append(np.full(len(TAIL_EPSILONS), np.nan))
            means.append(np.nan)
            variances.append(np.nan)
            continue
        values, tail, moments = _pack_moments(row, np.array(ALPHAS))
        walsh_from_spectrum.append(values)
        tails.append(tail)
        means.append(moments["mean_degree"])
        variances.append(moments["variance"])
    axes = np.zeros((len(BETAS), n, 3))
    axes[:, :, 0] = 1.0
    payload = dict(
        model="transverse_field_ising",
        n=n,
        boundary="open",
        params_json=json.dumps({"g": g}),
        method="open_boundary_free_fermion",
        betas=np.array(BETAS),
        alphas=np.array(ALPHAS),
        tail_epsilons=np.array(TAIL_EPSILONS),
        energies=model.energies,
        spectrum=spectrum,
        spectrum_computed=np.array(mask),
        W_fixed=np.array(walsh_trace),
        W_from_spectrum=np.array(walsh_from_spectrum),
        tails=np.array(tails),
        mean_degree=np.array(means),
        degree_variance=np.array(variances),
        mean_degree_error=np.array(mean_error),
        W_alpha0_intrinsic=np.array(alpha0),
        alpha0_certified_global=np.ones(len(BETAS), dtype=bool),
        alpha0_axes=axes,
        magnetization=np.stack(magnetization),
        energy=np.array(energy),
        energy_variance=np.array(variance),
        heat_capacity=np.array(heat),
        log_purity=np.array(log_purity),
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return f"wrote {name} in {time.time() - started:.1f}s spectra {int(np.sum(mask))}"


def _zz_job(n: int, boundary: str, out_dir: Path) -> str:
    started = time.time()
    name = case_name("commuting_zz_ising", n, boundary, {})
    destination = out_dir / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    spectra = []
    walsh_trace = []
    walsh_spectrum = []
    tails = []
    means = []
    variances = []
    mean_error = []
    energy = []
    for beta in BETAS:
        spectrum = classical_chain_spectrum(n, beta, boundary)
        spectra.append(spectrum)
        step = 1e-5
        plus = _classical_chain_log_generating_value(n, beta, np.exp(step), boundary).real
        minus = _classical_chain_log_generating_value(n, beta, np.exp(-step), boundary).real
        alpha0 = _centered_log_derivative(plus, minus, step)
        row = []
        for alpha in ALPHAS:
            if alpha == 0.0:
                row.append(alpha0)
            else:
                row.append(_classical_chain_log_generating_value(n, beta, np.exp(alpha), boundary).real / alpha)
        walsh_trace.append(row)
        values, tail, moments = _pack_moments(spectrum, np.array(ALPHAS, dtype=float))
        walsh_spectrum.append(values)
        tails.append(tail)
        means.append(moments["mean_degree"])
        variances.append(moments["variance"])
        mean_error.append(abs(moments["mean_degree"] - alpha0))
        bonds = n if boundary == "periodic" else n - 1
        energy.append(-bonds * float(np.tanh(beta)))
    axes = np.zeros((len(BETAS), n, 3))
    axes[:, :, 0] = 1.0
    # No field: every local Bloch vector of ρ² vanishes, so α→0 is basis-independent and equals n.
    payload = dict(
        model="commuting_zz_ising",
        n=n,
        boundary=boundary,
        params_json="{}",
        method="two_replica_transfer",
        betas=np.array(BETAS),
        alphas=np.array(ALPHAS),
        tail_epsilons=np.array(TAIL_EPSILONS),
        spectrum=np.stack(spectra),
        spectrum_computed=np.ones(len(BETAS), dtype=bool),
        W_fixed=np.array(walsh_trace),
        W_from_spectrum=np.array(walsh_spectrum),
        tails=np.array(tails),
        mean_degree=np.array(means),
        degree_variance=np.array(variances),
        mean_degree_error=np.array(mean_error),
        W_alpha0_intrinsic=np.full(len(BETAS), float(n)),
        alpha0_certified_global=np.ones(len(BETAS), dtype=bool),
        alpha0_axes=axes,
        energy=np.array(energy),
        reference_infinite_temperature=np.array([identity_walshness(n, alpha) for alpha in ALPHAS]),
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return f"wrote {name} in {time.time() - started:.1f}s max_mean_err {np.nanmax(mean_error):.3e}"


def _product_walshness(n: int, g: float, beta: float, alpha: float) -> float:
    """n independent copies. G = [(e^{2βg} + e^{-2βg} e^{2α}) / (2 cosh(2βg))]^n."""
    scale = 2.0 * float(g) * float(beta)
    if float(alpha) == 0.0:
        return float(n * (1.0 - np.tanh(scale)))
    log_moment = np.logaddexp(scale, -scale + 2.0 * float(alpha)) - np.logaddexp(scale, -scale)
    return float(n * log_moment / float(alpha))


def _product_job(n: int, g: float, out_dir: Path) -> str:
    started = time.time()
    name = case_name("one_site_product", n, "open", {"g": g})
    destination = out_dir / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    from operator_walshness.integrable import product_x_spectrum

    spectra = []
    walsh = []
    from_spectrum = []
    tails = []
    means = []
    variances = []
    mean_error = []
    for beta in BETAS:
        spectrum = product_x_spectrum(n, g, beta)
        spectra.append(spectrum)
        row = [_product_walshness(n, g, beta, alpha) for alpha in ALPHAS]
        walsh.append(row)
        values, tail, moments = _pack_moments(spectrum, np.array(ALPHAS, dtype=float))
        from_spectrum.append(values)
        tails.append(tail)
        means.append(moments["mean_degree"])
        variances.append(moments["variance"])
        mean_error.append(abs(moments["mean_degree"] - row[0]))
    magnetization = np.zeros((len(BETAS), n, 3))
    for index, beta in enumerate(BETAS):
        magnetization[index, :, 0] = np.tanh(beta * g)
    axes = np.zeros_like(magnetization)
    axes[:, :, 0] = 1.0
    payload = dict(
        model="one_site_product",
        n=n,
        boundary="open",
        params_json=json.dumps({"g": g}),
        method="exact_product",
        betas=np.array(BETAS),
        alphas=np.array(ALPHAS),
        tail_epsilons=np.array(TAIL_EPSILONS),
        spectrum=np.stack(spectra),
        spectrum_computed=np.ones(len(BETAS), dtype=bool),
        W_fixed=np.array(walsh),
        W_from_spectrum=np.array(from_spectrum),
        tails=np.array(tails),
        mean_degree=np.array(means),
        degree_variance=np.array(variances),
        mean_degree_error=np.array(mean_error),
        W_alpha0_intrinsic=np.array(walsh)[:, 0],
        alpha0_certified_global=np.ones(len(BETAS), dtype=bool),
        alpha0_axes=axes,
        magnetization=magnetization,
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return f"wrote {name} in {time.time() - started:.1f}s max_mean_err {np.nanmax(mean_error):.3e}"


def _dimer_job(n: int, out_dir: Path) -> str:
    started = time.time()
    name = case_name("disconnected_dimers", n, "open", {})
    destination = out_dir / f"{name}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {name}"
    if n % 2:
        raise ValueError("dimers need even n")
    evals, evecs, _ = diagonalize("disconnected_dimers", 2, "open", {})
    copies = n // 2
    spectra = []
    walsh = []
    from_spectrum = []
    tails = []
    means = []
    variances = []
    mean_error = []
    alpha0 = []
    axes = []
    for beta in BETAS:
        density, _ = gibbs_density(evals, evecs, beta)
        from operator_walshness.exact import gibbs_record

        one = gibbs_record(evals, evecs, beta)
        spectrum = convolve_copies(one["spectrum"], copies)
        spectra.append(spectrum)
        values, tail, moments = _pack_moments(one["spectrum"], np.array(ALPHAS, dtype=float))
        walsh.append(copies * values)
        big_values, big_tail, big_moments = _pack_moments(spectrum, np.array(ALPHAS, dtype=float))
        from_spectrum.append(big_values)
        tails.append(big_tail)
        means.append(big_moments["mean_degree"])
        variances.append(big_moments["variance"])
        mean_error.append(abs(big_moments["mean_degree"] - copies * values[0]))
        intrinsic, local_axes = intrinsic_alpha0(density)
        alpha0.append(copies * intrinsic)
        tiled = np.vstack([local_axes for _ in range(copies)])
        axes.append(tiled)
    payload = dict(
        model="disconnected_dimers",
        n=n,
        boundary="open",
        params_json="{}",
        method="exact_product_of_dimers",
        betas=np.array(BETAS),
        alphas=np.array(ALPHAS),
        tail_epsilons=np.array(TAIL_EPSILONS),
        spectrum=np.stack(spectra),
        spectrum_computed=np.ones(len(BETAS), dtype=bool),
        W_fixed=np.array(walsh),
        W_from_spectrum=np.array(from_spectrum),
        tails=np.array(tails),
        mean_degree=np.array(means),
        degree_variance=np.array(variances),
        mean_degree_error=np.array(mean_error),
        W_alpha0_intrinsic=np.array(alpha0),
        alpha0_certified_global=np.ones(len(BETAS), dtype=bool),
        alpha0_axes=np.stack(axes),
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return f"wrote {name} in {time.time() - started:.1f}s max_mean_err {np.nanmax(mean_error):.3e}"


def _dispatch(task):
    kind, n, extra, out = task
    out_dir = Path(out)
    if kind == "tfim":
        return _tfim_job(n, extra, out_dir)
    if kind == "zz":
        return _zz_job(n, extra, out_dir)
    if kind == "product":
        return _product_job(n, extra, out_dir)
    return _dimer_job(n, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/round1/integrable"))
    parser.add_argument("--sizes", type=int, nargs="*", default=[32, 64, 128, 256, 512])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    for n in args.sizes:
        for g in (0.5, 0.8, 1.0, 1.2, 1.5):
            tasks.append(("tfim", n, g, str(args.out)))
        for boundary in ("open", "periodic"):
            tasks.append(("zz", n, boundary, str(args.out)))
        for g in (0.5, 1.0):
            tasks.append(("product", n, g, str(args.out)))
        if n % 2 == 0:
            tasks.append(("dimer", n, None, str(args.out)))
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "code_hash": _code_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tasks": len(tasks),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.workers == 1:
        for task in tasks:
            print(_dispatch(task), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_dispatch, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
