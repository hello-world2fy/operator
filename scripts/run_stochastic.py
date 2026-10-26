"""Matrix-free Hutchinson estimate of fixed-basis Operator Walshness.

The Hamiltonian is applied on the fly, so n=16–22 does not store a dense
matrix. Real Rademacher probes estimate G(e^α). The same probes enter the
numerator and the denominator. A second seed is a held-out repeat, not a
basis optimization. This does not produce P(k).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.free_fermion import OpenBoundaryTFIM
from operator_walshness.grid import BETAS, case_name
from operator_walshness.stochastic import apply_sqrt_a

ALPHAS = (0.5, 1.0, 2.0)


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "operator_walshness"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _apply(vector: np.ndarray, n: int, model: str, g: float, h: float, delta: float) -> np.ndarray:
    shape = vector.shape
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    idx = np.arange(vector.size)
    out = np.zeros(vector.shape, dtype=np.float64)
    model = model.lower()
    if model in {"transverse_field_ising", "tfim", "mixed_field_ising", "mixed", "commuting_zz_ising", "zz"}:
        for site in range(n - 1):
            zi = 1 - 2 * ((idx >> site) & 1)
            zj = 1 - 2 * ((idx >> (site + 1)) & 1)
            out += -zi * zj * vector
        if model not in {"commuting_zz_ising", "zz"}:
            for site in range(n):
                out += -g * vector[idx ^ (1 << site)]
        if model in {"mixed_field_ising", "mixed"}:
            for site in range(n):
                bit = (idx >> site) & 1
                out += -h * (1 - 2 * bit) * vector
        return out.reshape(shape)
    if model in {"xxz"}:
        for site in range(n - 1):
            flipped = vector[idx ^ (1 << site) ^ (1 << (site + 1))]
            bi = (idx >> site) & 1
            bj = (idx >> (site + 1)) & 1
            opposite = bi != bj
            out += np.where(opposite, 2.0 * flipped, 0.0)
            zi = 1 - 2 * bi
            zj = 1 - 2 * bj
            out += delta * zi * zj * vector
        return out.reshape(shape)
    raise ValueError(f"no matrix-free action for {model}")


def _sparse_hamiltonian(n: int, model: str, g: float, h: float, delta: float) -> csr_matrix:
    """Same real-symmetric action as ``_apply``, stored once."""
    dimension = 1 << n
    index = np.arange(dimension)
    diagonal = np.zeros(dimension, dtype=np.float64)
    model = model.lower()
    if model in {"transverse_field_ising", "tfim", "mixed_field_ising", "mixed", "commuting_zz_ising", "zz"}:
        for site in range(n - 1):
            zi = 1 - 2 * ((index >> site) & 1)
            zj = 1 - 2 * ((index >> (site + 1)) & 1)
            diagonal += -zi * zj
        if model in {"mixed_field_ising", "mixed"}:
            for site in range(n):
                diagonal += -h * (1 - 2 * ((index >> site) & 1))
        rows = [index]
        cols = [index]
        data = [diagonal]
        if model not in {"commuting_zz_ising", "zz"}:
            for site in range(n):
                low = index[(index & (1 << site)) == 0]
                high = low ^ (1 << site)
                coupling = np.full(low.shape, -g)
                rows.extend((low, high))
                cols.extend((high, low))
                data.extend((coupling, coupling))
        return csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(dimension, dimension))
    if model == "xxz":
        for site in range(n - 1):
            zi = 1 - 2 * ((index >> site) & 1)
            zj = 1 - 2 * ((index >> (site + 1)) & 1)
            diagonal += delta * zi * zj
        rows = [index]
        cols = [index]
        data = [diagonal]
        for site in range(n - 1):
            low = index[(index & (1 << site)) == 0]
            differ = (low & (1 << (site + 1))) != 0
            src = low[differ]
            dst = src ^ (1 << site) ^ (1 << (site + 1))
            coupling = np.full(src.shape, 2.0)
            rows.extend((src, dst))
            cols.extend((dst, src))
            data.extend((coupling, coupling))
        return csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(dimension, dimension))
    raise ValueError(f"no matrix-free action for {model}")


def _estimate(n: int, model: str, params: dict, beta: float, alpha: float, probes: int, seed: int) -> dict[str, float]:
    g = float(params.get("g", 1.0))
    h = float(params.get("h", 0.0))
    delta = float(params.get("delta", 1.0))
    dimension = 1 << n
    operator = -float(beta) * _sparse_hamiltonian(n, model, g, h, delta)
    batches = 4
    per_batch = probes // batches
    rng = np.random.default_rng(seed)
    numerators = []
    denominators = []
    for _ in range(batches):
        num = 0.0
        den = 0.0
        for _probe in range(per_batch):
            probe = rng.choice(np.array([-1.0, 1.0]), size=dimension)
            evolved = expm_multiply(operator, probe, traceA=0.0)
            den += float(np.dot(evolved, evolved))
            twisted = apply_sqrt_a(probe, alpha, n)
            evolved_twist = expm_multiply(operator, twisted, traceA=0.0)
            finished = apply_sqrt_a(evolved_twist, alpha, n)
            num += float(np.dot(finished, finished))
        numerators.append(num)
        denominators.append(den)
    ratios = np.array(numerators) / np.array(denominators)
    ratio = float(np.sum(numerators) / np.sum(denominators))
    stderr = float(np.std(ratios, ddof=1) / np.sqrt(batches))
    return {
        "W": float(np.log(ratio) / alpha),
        "moment": ratio,
        "moment_stderr": stderr,
        "W_stderr": float(stderr / (abs(ratio) * alpha)),
    }


def _dispatch(task: tuple) -> str:
    model, params, n, beta, alpha, probes, out = task
    started = time.time()
    name = case_name(model, n, "open", params)
    tag = f"{name}_b{beta:.6g}_a{alpha:.6g}_p{probes}"
    destination = Path(out) / f"{tag}.npz"
    if destination.exists() and os.environ.get("WALSH_OVERWRITE", "0") != "1":
        return f"skip {tag}"
    primary = _estimate(n, model, params, beta, alpha, probes, seed=0)
    held = _estimate(n, model, params, beta, alpha, probes, seed=1)
    reference = np.nan
    if model == "transverse_field_ising":
        reference = OpenBoundaryTFIM(n, float(params["g"])).fixed_walshness(beta, alpha)
    payload = dict(
        model=model,
        n=n,
        boundary="open",
        params_json=json.dumps(params),
        method="hutchinson_matrix_free",
        beta=float(beta),
        alpha=float(alpha),
        probes=int(probes),
        batches=4,
        W=primary["W"],
        moment=primary["moment"],
        moment_stderr=primary["moment_stderr"],
        W_stderr=primary["W_stderr"],
        heldout_W=held["W"],
        heldout_W_stderr=held["W_stderr"],
        reference_W=reference,
        reference_method="free_fermion" if np.isfinite(reference) else "none",
        W_abs_error=float(abs(primary["W"] - reference)) if np.isfinite(reference) else np.nan,
        code_hash=_code_hash(),
        runtime_seconds=time.time() - started,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return (
        f"wrote {tag} in {payload['runtime_seconds']:.1f}s "
        f"W {primary['W']:.6f} ± {primary['W_stderr']:.3e} "
        f"held {held['W']:.6f} ref_err {payload['W_abs_error']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/round1/stochastic"))
    parser.add_argument("--sizes", type=int, nargs="*", default=[16])
    parser.add_argument("--probes", type=int, nargs="*", default=[32, 128, 512])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--betas", type=float, nargs="*", default=list(BETAS))
    args = parser.parse_args()
    cases = [
        ("transverse_field_ising", {"g": 1.0}),
        ("mixed_field_ising", {"g": 1.05, "h": 0.5}),
    ]
    tasks = []
    for n in args.sizes:
        probe_counts = [count for count in args.probes if n <= 16 or count <= 128]
        if n >= 20:
            probe_counts = [count for count in probe_counts if count <= 32]
        for model, params in cases:
            for beta in args.betas:
                for alpha in ALPHAS:
                    for probes in probe_counts:
                        tasks.append((model, params, n, float(beta), float(alpha), int(probes), str(args.out)))
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.json"
    if meta_path.exists():
        tag = "-".join(str(size) for size in args.sizes)
        meta_path = args.out / f"metadata_n{tag}.json"
    meta_path.write_text(
        json.dumps({"code_hash": _code_hash(), "tasks": len(tasks), "note": "fixed computational basis only"}, indent=2)
    )
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
