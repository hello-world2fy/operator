"""Screen |w(χ) - w(2χ)| for saved MPO files. This is not an error bar."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "round1" / "mpo"


def main() -> None:
    grouped: dict[tuple[str, float], dict[int, np.ndarray]] = {}
    discarded: dict[tuple[str, float, int], float] = {}
    for path in sorted(ROOT.glob("*_dt0.05_b*.npz")):
        if "_chi" not in path.name:
            continue
        data = np.load(path)
        if float(data["dt_requested"]) != 0.05:
            continue
        key = (path.name.split("_chi")[0], float(data["beta"]))
        chi = int(data["chi_max"])
        grouped.setdefault(key, {})[chi] = np.array(data["W_log"], dtype=float)
        discarded[(key[0], key[1], chi)] = float(data["discarded_weight"])
    print(f"{'case':<42} {'beta':>6} {'|dw|max':>10} {'|dw|_1':>10} {'disc64':>9} {'disc128':>9}")
    worst = 0.0
    count = 0
    for (case, beta), by_chi in sorted(grouped.items()):
        if 64 not in by_chi or 128 not in by_chi:
            continue
        n = int(case.split("_n")[1].split("_")[0])
        delta = np.abs(by_chi[128] - by_chi[64]) / (2 * n)
        count += 1
        worst = max(worst, float(np.max(delta)))
        if float(np.max(delta)) < 1e-3 and beta not in {1.0, 4.0, 16.0}:
            continue
        print(
            f"{case:<42} {beta:6g} {float(np.max(delta)):10.3e} {float(delta[3]):10.3e} "
            f"{discarded[(case, beta, 64)]:9.2e} {discarded[(case, beta, 128)]:9.2e}"
        )
    print(f"pairs {count} worst density gap {worst:.3e}")


if __name__ == "__main__":
    main()
