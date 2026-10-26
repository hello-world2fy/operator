"""Selective n=14 exact diagonalization on one GPU.

Transverse-field Ising and one mixed-field point keep the Gibbs grid, the
eigenstate table and the microcanonical windows. Classical ZZ keeps the Gibbs
grid only: its spectrum is degenerate, so individual eigenvectors are not
unique. Numerical output matches the n<=12 census schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.exact import eigenstate_table, gibbs_record, microcanonical_table
from operator_walshness.free_fermion import OpenBoundaryTFIM
from operator_walshness.grid import ALPHAS, BETAS, TAIL_EPSILONS, case_name
from operator_walshness.hamiltonians import build_hamiltonian
from operator_walshness.integrable import classical_chain_walshness

CASES = (
    ("transverse_field_ising", {"g": 1.0}, True),
    ("mixed_field_ising", {"g": 1.05, "h": 0.5}, True),
    ("commuting_zz_ising", {}, False),
)


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _diagonalize(model: str, params: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = build_hamiltonian(
        model,
        14,
        g=float(params.get("g", 1.0)),
        h=float(params.get("h", 0.0)),
        delta=float(params.get("delta", 1.0)),
        boundary="open",
    )
    matrix = torch.tensor(hamiltonian, dtype=torch.float64, device=device)
    del hamiltonian
    evals, evecs = torch.linalg.eigh(matrix)
    del matrix
    torch.cuda.empty_cache()
    return evals.detach().cpu().numpy(), evecs.detach().cpu().numpy()


def _reference_error(model: str, params: dict, beta: float, walsh: np.ndarray) -> float:
    if model == "transverse_field_ising":
        fermion = OpenBoundaryTFIM(14, float(params["g"]))
        reference = np.array([fermion.fixed_walshness(beta, alpha) for alpha in ALPHAS])
        return float(np.max(np.abs(reference - walsh)))
    if model == "commuting_zz_ising":
        reference = np.array([classical_chain_walshness(14, beta, alpha, "open") for alpha in ALPHAS])
        return float(np.max(np.abs(reference - walsh)))
    return float("nan")


def main() -> None:
    device = os.environ.get("WALSH_DEVICE", "cuda:4")
    out = Path(__file__).resolve().parents[1] / "results" / "round1" / "exact"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for model, params, excited in CASES:
        name = case_name(model, 14, "open", params)
        destination = out / f"{name}.npz"
        if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
            print(f"skip {name}", flush=True)
            continue
        started = time.time()
        print(f"diagonalizing {name} on {device}", flush=True)
        evals, evecs = _diagonalize(model, params, device)
        print(f"eigh {name} in {time.time() - started:.1f}s", flush=True)
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
        reference_error = []
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
            reference_error.append(_reference_error(model, params, beta, record["W_fixed"]))
            schmidt_chunks.append(record["schmidt"])
            print(
                f"  {name} beta {beta:g} W1 {record['W_fixed'][3]:.6f} "
                f"ref_err {reference_error[-1]:.3e} in {time.time() - started:.1f}s",
                flush=True,
            )
        payload = dict(
            model=model,
            n=14,
            boundary="open",
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
            reference_abs_error=np.array(reference_error),
            code_hash=_code_hash(),
            runtime_seconds=time.time() - started,
        )
        if excited:
            payload.update({f"eigen_{key}": value for key, value in eigenstate_table(evals, evecs, np.full(evals.size, np.nan), ALPHAS).items()})
            print(f"eigenstates {name} in {time.time() - started:.1f}s", flush=True)
            payload.update({f"micro_{key}": value for key, value in microcanonical_table(evals, evecs, ALPHAS).items()})
            print(f"microcanonical {name} in {time.time() - started:.1f}s", flush=True)
        payload["schmidt"] = np.array(schmidt_chunks, dtype=object)
        np.savez_compressed(destination, **payload)
        print(
            f"wrote {name} in {time.time() - started:.1f}s "
            f"max_ref_err {np.nanmax(reference_error):.3e}",
            flush=True,
        )
        del evals, evecs, payload


if __name__ == "__main__":
    main()
