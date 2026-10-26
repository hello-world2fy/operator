"""Round-1 model and parameter grid, taken from the A100 run plan."""

from __future__ import annotations

BETAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
EXACT_SIZES = (6, 8, 10, 12)
TAIL_EPSILONS = (1e-2, 1e-3, 1e-4, 1e-6)

# (model, boundary, parameter dict). Parameters use the symbol in the run plan.
EXACT_CASES = []
for g in (0.5, 1.0):
    EXACT_CASES.append(("one_site_product", "open", {"g": g}))
EXACT_CASES.append(("disconnected_dimers", "open", {}))
EXACT_CASES.append(("commuting_zz_ising", "open", {}))
EXACT_CASES.append(("commuting_zz_ising", "periodic", {}))
for g in (0.5, 0.8, 1.0, 1.2, 1.5):
    EXACT_CASES.append(("transverse_field_ising", "open", {"g": g}))
EXACT_CASES.append(("transverse_field_ising", "periodic", {"g": 1.0}))
for g, h in ((0.8, 0.2), (1.05, 0.2), (1.05, 0.5), (1.2, 0.5)):
    EXACT_CASES.append(("mixed_field_ising", "open", {"g": g, "h": h}))
for delta in (0.5, 1.0):
    EXACT_CASES.append(("xxz", "open", {"delta": delta}))

EXACT_2D = ((2, 2), (3, 2), (4, 2), (3, 3), (4, 3))

EXCITED_MODELS = {"transverse_field_ising", "mixed_field_ising"}
MICROCANONICAL_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
MICROCANONICAL_RANKS = (1, 2, 4, 8, 16, 32)


def case_name(model: str, n: int, boundary: str, params: dict) -> str:
    parts = [model, f"n{n}", boundary]
    for key in ("g", "h", "delta", "lx", "ly"):
        if key in params:
            parts.append(f"{key}{params[key]:g}")
    return "_".join(parts)
