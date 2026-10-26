"""Exact diagonalization census for Operator Walshness.

One diagonalization serves every temperature. Fixed-basis spectra are stored
in full. α→0 intrinsic values are the closed-form magnetization result, not a
numerical upper bound.
"""

from __future__ import annotations

import numpy as np

from .grid import ALPHAS, MICROCANONICAL_FRACTIONS, MICROCANONICAL_RANKS, TAIL_EPSILONS
from .hamiltonians import build_hamiltonian, ising_2d_hamiltonian, xxz_sector_hamiltonian
from .observables import (
    degree_correlations_from_x_matrix,
    distance_correlations,
    ensemble_magnetization,
    operator_schmidt,
    parity_expectation,
    state_half_entropy,
    thermal_observables,
    total_sz_expectation,
)
from .walsh import (
    degree_histogram,
    fixed_basis_spectrum,
    fixed_x_directions,
    gibbs_density,
    spectrum_from_masses,
    spectrum_moments,
    state_degree_spectrum,
    tail_degree,
    walshness_from_directions,
    walshness_from_spectrum,
    x_basis_matrix,
)


def diagonalize(model: str, n: int, boundary: str, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eigenvalues, eigenvectors (columns), and a sector label per state.

    Sector label is total S^z for XXZ and NaN otherwise. XXZ is diagonalized
    inside magnetization sectors so the eigenvectors are not mixed by degeneracy
    between different charge sectors.
    """
    if model == "xxz":
        blocks_e = []
        blocks_v = []
        labels = []
        delta = float(params.get("delta", 1.0))
        for down in range(n + 1):
            block, basis = xxz_sector_hamiltonian(n, down, delta=delta, boundary=boundary)
            evals, evecs = np.linalg.eigh(block)
            full = np.zeros((1 << n, evals.size), dtype=np.float64)
            full[basis, :] = evecs
            blocks_e.append(evals)
            blocks_v.append(full)
            labels.append(np.full(evals.size, 0.5 * (n - 2 * down)))
        evals = np.concatenate(blocks_e)
        evecs = np.concatenate(blocks_v, axis=1)
        sector = np.concatenate(labels)
        order = np.argsort(evals, kind="mergesort")
        return evals[order], evecs[:, order], sector[order]
    if model == "ising_2d_classical":
        hamiltonian = ising_2d_hamiltonian(int(params["lx"]), int(params["ly"]))
    else:
        hamiltonian = build_hamiltonian(
            model,
            n,
            g=float(params.get("g", 1.0)),
            h=float(params.get("h", 0.0)),
            delta=float(params.get("delta", 1.0)),
            boundary=boundary,
        )
    evals, evecs = np.linalg.eigh(hamiltonian)
    return evals, evecs, np.full(evals.size, np.nan)


def degeneracy_groups(evals: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    labels = np.zeros(evals.size, dtype=np.int32)
    group = 0
    for index in range(1, evals.size):
        scale = max(1.0, abs(float(evals[index])))
        if abs(float(evals[index] - evals[index - 1])) > tol * scale:
            group += 1
        labels[index] = group
    return labels


def matched_beta(evals: np.ndarray, target_energy: float) -> float:
    """Inverse temperature, positive or negative, matching a canonical mean energy."""
    e_inf = float(np.mean(evals))
    if abs(target_energy - e_inf) <= 1e-10 * max(1.0, abs(e_inf)):
        return 0.0

    def mean_energy(beta: float) -> float:
        logits = -beta * evals
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        weights /= weights.sum()
        return float(np.dot(weights, evals))

    # Mean energy decreases with β.
    sign = 1.0 if target_energy < e_inf else -1.0
    lo, hi = 0.0, 1.0
    while sign * mean_energy(sign * hi) > sign * target_energy and hi < 1e6:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sign * mean_energy(sign * mid) > sign * target_energy:
            lo = mid
        else:
            hi = mid
    return float(sign * 0.5 * (lo + hi))


def _spectrum_record(spectrum: np.ndarray, alphas: tuple[float, ...]) -> dict[str, np.ndarray | float]:
    moments = spectrum_moments(spectrum)
    walsh = np.array([walshness_from_spectrum(spectrum, alpha) for alpha in alphas])
    tails = np.array([tail_degree(spectrum, eps) for eps in TAIL_EPSILONS])
    return {"W": walsh, "tails": tails, **moments}


def gibbs_record(evals: np.ndarray, evecs: np.ndarray, beta: float, alphas: tuple[float, ...] = ALPHAS) -> dict:
    density, probabilities = gibbs_density(evals, evecs, beta)
    x_matrix = x_basis_matrix(density)
    n = int(np.log2(density.shape[0]))
    masses, mass_sum = degree_histogram(x_matrix, n)
    spectrum = spectrum_from_masses(masses)
    fro = float(np.vdot(density, density).real)
    diagnostics = {
        "mass_sum": mass_sum,
        "frobenius_squared": fro,
        "mass_relative_error": abs(mass_sum - fro) / max(fro, 1e-300),
        "probability_defect": abs(float(spectrum.sum()) - 1.0),
        "min_bin": float(np.min(spectrum)),
    }
    record = _spectrum_record(spectrum, alphas)
    thermo = thermal_observables(evals, probabilities, beta)
    profile = ensemble_magnetization(evecs, probabilities)
    doubled_probabilities = gibbs_density(evals, evecs, 2.0 * float(beta))[1]
    doubled_profile = ensemble_magnetization(evecs, doubled_probabilities)
    lengths = np.linalg.norm(doubled_profile, axis=1)
    intrinsic = float(np.sum(1.0 - lengths))
    axes = np.zeros_like(doubled_profile)
    for site, length in enumerate(lengths):
        axes[site] = doubled_profile[site] / length if length >= 1e-14 else np.array([1.0, 0.0, 0.0])
    directions = fixed_x_directions(n)
    predicted = walshness_from_directions(density, directions, 1.0)
    alpha_index = list(alphas).index(1.0) if 1.0 in alphas else None
    product_errors = np.zeros(len(alphas))
    if alpha_index is not None:
        product_errors[alpha_index] = abs(predicted - record["W"][alpha_index]) / max(abs(predicted), 1e-300)
    schmidt, entropy = operator_schmidt(density)
    degree = degree_correlations_from_x_matrix(x_matrix)
    return {
        "spectrum": spectrum,
        "diagnostics": diagnostics,
        "W_fixed": record["W"],
        "tails": record["tails"],
        "moments": {key: record[key] for key in ("mean_degree", "variance", "second_moment", "third_central")},
        "thermo": thermo,
        "magnetization": profile,
        "W_alpha0_intrinsic": intrinsic,
        "alpha0_axes": axes,
        "product_relative_error": np.array(product_errors),
        "osee": entropy,
        "schmidt": schmidt[schmidt > 1e-14],
        "correlation_x": distance_correlations(density, "X"),
        "correlation_z": distance_correlations(density, "Z"),
        "degree_mean": degree["degree_mean"],
        "degree_correlation": degree["degree_correlation"],
        "density": density,
        "probabilities": probabilities,
    }


def eigenstate_table(evals: np.ndarray, evecs: np.ndarray, sector: np.ndarray, alphas: tuple[float, ...] = ALPHAS) -> dict[str, np.ndarray]:
    count, n_states = evecs.shape[1], evecs.shape[1]
    n = int(np.log2(evecs.shape[0]))
    parity = np.empty(n_states)
    sz = np.empty(n_states)
    entropy = np.empty(n_states)
    state_p = np.zeros((n_states, n + 1))
    walsh = np.zeros((n_states, len(alphas)))
    for index in range(n_states):
        state = evecs[:, index]
        parity[index] = parity_expectation(state)
        sz[index] = float(sector[index]) if np.isfinite(sector[index]) else total_sz_expectation(state)
        _, entropy[index] = state_half_entropy(state)
        spectrum = state_degree_spectrum(state)
        state_p[index] = spectrum
        # Operator Walshness of a pure projector is exactly twice the state value.
        walsh[index] = [2.0 * walshness_from_spectrum(spectrum, alpha) for alpha in alphas]
    groups = degeneracy_groups(evals)
    return {
        "energy": evals,
        "parity": parity,
        "sz": sz,
        "entropy": entropy,
        "state_spectrum": state_p,
        "W_operator": walsh,
        "degeneracy_group": groups,
    }


def microcanonical_table(
    evals: np.ndarray,
    evecs: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS,
) -> dict[str, np.ndarray]:
    n_states = evals.size
    rows = []
    for fraction in MICROCANONICAL_FRACTIONS:
        center = int(round(fraction * (n_states - 1)))
        for rank in MICROCANONICAL_RANKS:
            if rank > n_states:
                continue
            start = min(max(center - rank // 2, 0), n_states - rank)
            window = slice(start, start + rank)
            vectors = evecs[:, window]
            density = (vectors @ vectors.T) / rank
            spectrum, _ = fixed_basis_spectrum(density)
            packed = _spectrum_record(spectrum, alphas)
            mean_energy = float(np.mean(evals[window]))
            beta = matched_beta(evals, mean_energy)
            gibbs_beta = float(np.clip(beta, -1e5, 1e5))
            gibbs, gibbs_prob = gibbs_density(evals, evecs, gibbs_beta)
            gibbs_spectrum, _ = fixed_basis_spectrum(gibbs)
            gibbs_w = np.array([walshness_from_spectrum(gibbs_spectrum, alpha) for alpha in alphas])
            edge_gap = 0.0
            if start > 0:
                edge_gap = max(edge_gap, abs(float(evals[start] - evals[start - 1])))
            if start + rank < n_states:
                edge_gap = max(edge_gap, abs(float(evals[start + rank] - evals[start + rank - 1])))
            rows.append(
                {
                    "fraction": fraction,
                    "rank": rank,
                    "start": start,
                    "mean_energy": mean_energy,
                    "edge_gap": edge_gap,
                    "purity": 1.0 / rank,
                    "beta_matched": beta,
                    "gibbs_energy": float(np.dot(gibbs_prob, evals)),
                    "spectrum": spectrum,
                    "W": packed["W"],
                    "tails": packed["tails"],
                    "mean_degree": packed["mean_degree"],
                    "variance": packed["variance"],
                    "gibbs_W": gibbs_w,
                    "W_alpha0_intrinsic": float(np.sum(1.0 - np.linalg.norm(ensemble_magnetization(vectors, np.full(rank, 1.0 / rank)), axis=1))),
                    "gibbs_W_alpha0_intrinsic": float(
                        np.sum(1.0 - np.linalg.norm(ensemble_magnetization(evecs, gibbs_density(evals, evecs, 2.0 * gibbs_beta)[1]), axis=1))
                    ),
                }
            )
    count = len(rows)
    degree = rows[0]["spectrum"].size
    n_alpha = len(alphas)
    out = {
        "fraction": np.array([row["fraction"] for row in rows]),
        "rank": np.array([row["rank"] for row in rows], dtype=np.int32),
        "start": np.array([row["start"] for row in rows], dtype=np.int32),
        "mean_energy": np.array([row["mean_energy"] for row in rows]),
        "edge_gap": np.array([row["edge_gap"] for row in rows]),
        "purity": np.array([row["purity"] for row in rows]),
        "beta_matched": np.array([row["beta_matched"] for row in rows]),
        "gibbs_energy": np.array([row["gibbs_energy"] for row in rows]),
        "spectrum": np.stack([row["spectrum"] for row in rows]),
        "W": np.stack([row["W"] for row in rows]),
        "tails": np.stack([row["tails"] for row in rows]),
        "mean_degree": np.array([row["mean_degree"] for row in rows]),
        "variance": np.array([row["variance"] for row in rows]),
        "gibbs_W": np.stack([row["gibbs_W"] for row in rows]),
        "W_alpha0_intrinsic": np.array([row["W_alpha0_intrinsic"] for row in rows]),
        "gibbs_W_alpha0_intrinsic": np.array([row["gibbs_W_alpha0_intrinsic"] for row in rows]),
    }
    if out["spectrum"].shape != (count, degree) or out["W"].shape != (count, n_alpha):
        raise RuntimeError("microcanonical table shape mismatch")
    return out
