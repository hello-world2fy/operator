"""Fixed-X Walshness of XXZ eigenstates, one magnetization sector at a time.

XXZ conserves the number of down spins. The saved exact files store the Gibbs
operator, not these eigenvectors. A single eigenstate's operator Walshness is
twice its state Walshness. Values inside a degenerate multiplet depend on the
basis chosen by the eigensolver, so the summary uses only levels that are
unique inside their sector.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_walshness.grid import ALPHAS
from operator_walshness.hamiltonians import xxz_sector_hamiltonian
from operator_walshness.observables import state_half_entropy
from operator_walshness.walsh import fwht, hamming_weights, state_walshness


def _operator_walshness(spectrum: np.ndarray, alphas: tuple[float, ...]) -> np.ndarray:
    degrees = np.arange(spectrum.shape[1], dtype=np.float64)
    out = np.empty((spectrum.shape[0], len(alphas)), dtype=np.float64)
    for column, alpha in enumerate(alphas):
        if alpha == 0.0:
            out[:, column] = 2.0 * spectrum @ degrees
            continue
        scores = np.full(spectrum.shape, -np.inf, dtype=np.float64)
        positive = spectrum > 0.0
        scores[positive] = np.log(spectrum[positive]) + alpha * degrees[np.nonzero(positive)[1]]
        shift = np.max(scores, axis=1, keepdims=True)
        log_moment = shift[:, 0] + np.log(np.sum(np.exp(scores - shift), axis=1))
        out[:, column] = 2.0 * log_moment / alpha
    return out


def _unique_mask(energy: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    order = np.argsort(energy, kind="mergesort")
    sorted_energy = energy[order]
    gaps = np.diff(sorted_energy)
    close = gaps <= tol
    grouped = np.zeros(energy.size, dtype=bool)
    grouped[1:] |= close
    grouped[:-1] |= close
    unique_sorted = ~grouped
    unique = np.empty(energy.size, dtype=bool)
    unique[order] = unique_sorted
    return unique


def scan_case(n: int, delta: float, boundary: str, alphas: tuple[float, ...] = ALPHAS) -> dict[str, np.ndarray]:
    pieces: list[dict[str, np.ndarray]] = []
    for n_down in range(n + 1):
        started = time.perf_counter()
        block, basis = xxz_sector_hamiltonian(n, n_down, delta=delta, boundary=boundary)
        energy, vectors = np.linalg.eigh(block)
        embedded = np.zeros((vectors.shape[1], 1 << n), dtype=np.complex128)
        embedded[:, basis] = vectors.T
        amplitudes = fwht(embedded, axis=1) / np.sqrt(1 << n)
        probabilities = np.abs(amplitudes) ** 2
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        weights = hamming_weights(n)
        spectrum = np.stack(
            [probabilities[:, weights == degree].sum(axis=1) for degree in range(n + 1)],
            axis=1,
        )
        check = 2.0 * state_walshness(embedded[0], 1.0)
        batch = float(_operator_walshness(spectrum[:1], (1.0,))[0, 0])
        if abs(check - batch) > 1e-9:
            raise RuntimeError(f"sector Walshness mismatch {check} vs {batch}")
        entropy = np.empty(energy.size, dtype=np.float64)
        for index in range(energy.size):
            _, entropy[index] = state_half_entropy(embedded[index])
        pieces.append(
            {
                "energy": energy,
                "n_down": np.full(energy.size, n_down, dtype=np.int16),
                "entropy": entropy,
                "spectrum": spectrum,
                "W_operator": _operator_walshness(spectrum, alphas),
                "unique": _unique_mask(energy),
            }
        )
        print(
            f"  n={n} delta={delta} {boundary} down={n_down} dim={energy.size} "
            f"unique={int(pieces[-1]['unique'].sum())} in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    return {key: np.concatenate([item[key] for item in pieces]) for key in pieces[0]}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "results" / "round1" / "xxz_sectors"
    destination.mkdir(parents=True, exist_ok=True)
    for n in (6, 8, 10, 12):
        for delta in (0.5, 1.0):
            name = f"xxz_n{n}_open_delta{delta:g}.npz"
            path = destination / name
            if path.exists():
                print(f"skip {name}", flush=True)
                continue
            started = time.perf_counter()
            data = scan_case(n, delta, "open")
            half = data["n_down"] == n // 2
            unique = data["unique"] & half
            alpha_index = list(ALPHAS).index(1.0)
            w1 = data["W_operator"][:, alpha_index]
            np.savez_compressed(
                path,
                model="xxz",
                n=n,
                boundary="open",
                params_json=json.dumps({"delta": delta}),
                alphas=np.asarray(ALPHAS, dtype=np.float64),
                energy=data["energy"],
                n_down=data["n_down"],
                entropy=data["entropy"],
                state_spectrum=data["spectrum"],
                W_operator=data["W_operator"],
                unique_in_sector=data["unique"],
                convention="fixed computational X basis; operator W = 2 * state W; degenerate eigenvectors are basis-dependent",
            )
            ground = np.flatnonzero(half)[np.argmin(data["energy"][half])]
            print(
                f"wrote {name} in {time.perf_counter() - started:.1f}s "
                f"half-filling unique {int(unique.sum())}/{int(half.sum())} "
                f"ground W1 {w1[ground]:.6f} density {w1[ground] / (2 * n):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
