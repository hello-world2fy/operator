"""Repeat a few n=8 points with seeds 1, 2 and 3.

The main campaign uses seed 0. These repeats check whether a different set of
random starts finds a lower upper bound. They are not certified minima.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_basis_opt import _one_point

CASES = (
    ("transverse_field_ising", "open", {"g": 1.0}),
    ("mixed_field_ising", "open", {"g": 0.8, "h": 0.2}),
)


def main() -> None:
    devices = ["cuda:5", "cuda:6", "cuda:7"]
    out = str(Path(__file__).resolve().parents[1] / "results" / "round1" / "basis")
    tasks = []
    for seed in (1, 2, 3):
        for beta in (1.0, 4.0):
            for model, boundary, params in CASES:
                device = devices[len(tasks) % len(devices)]
                tasks.append((model, 8, boundary, params, beta, 1.0, seed, 24, device, out))
    print(f"tasks {len(tasks)}", flush=True)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
        futures = [pool.submit(_one_point, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
