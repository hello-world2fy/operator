"""Spin Hamiltonians for the round-1 Operator Walshness census.

Site ``i`` is bit ``i`` of the computational basis integer (site 0 is least
significant). Pauli matrices are the standard textbook matrices. Every
Hamiltonian below is copied from the round-1 run plan without rescaling.
"""

from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=np.float64)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def check_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer")
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    return n


def bonds(n: int, boundary: str) -> list[tuple[int, int]]:
    boundary = boundary.lower()
    if boundary not in {"open", "obc", "periodic", "pbc"}:
        raise ValueError("boundary must be open or periodic")
    pairs = [(i, i + 1) for i in range(n - 1)]
    if boundary in {"periodic", "pbc"}:
        if n < 2:
            raise ValueError("periodic boundary needs n>=2")
        pairs.append((n - 1, 0))
    return pairs


def _zeros(n: int) -> np.ndarray:
    return np.zeros((1 << n, 1 << n), dtype=np.float64)


def add_x(h: np.ndarray, site: int, coeff: float = 1.0) -> None:
    n = int(np.log2(h.shape[0]))
    idx = np.arange(h.shape[0])
    h[idx ^ (1 << site), idx] += coeff


def add_z(h: np.ndarray, site: int, coeff: float = 1.0) -> None:
    idx = np.arange(h.shape[0])
    bit = (idx >> site) & 1
    h[idx, idx] += coeff * (1 - 2 * bit)


def add_zz(h: np.ndarray, i: int, j: int, coeff: float = 1.0) -> None:
    idx = np.arange(h.shape[0])
    zi = 1 - 2 * ((idx >> i) & 1)
    zj = 1 - 2 * ((idx >> j) & 1)
    h[idx, idx] += coeff * zi * zj


def add_xx(h: np.ndarray, i: int, j: int, coeff: float = 1.0) -> None:
    idx = np.arange(h.shape[0])
    h[idx ^ (1 << i) ^ (1 << j), idx] += coeff


def add_yy(h: np.ndarray, i: int, j: int, coeff: float = 1.0) -> None:
    """Y_i Y_j is real: the two factors of i multiply to -1."""
    idx = np.arange(h.shape[0])
    bi = (idx >> i) & 1
    bj = (idx >> j) & 1
    phase = -((1 - 2 * bi) * (1 - 2 * bj))
    h[idx ^ (1 << i) ^ (1 << j), idx] += coeff * phase


def pauli_embedding(n: int, pauli: str, site: int) -> np.ndarray:
    """Dense embedding of one local Pauli, used by tests and expectations."""
    n = check_n(n)
    h = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    if pauli == "I":
        np.fill_diagonal(h, 1.0)
    elif pauli == "X":
        add_x(h, site, 1.0)
    elif pauli == "Z":
        add_z(h, site, 1.0)
    elif pauli == "Y":
        idx = np.arange(1 << n)
        bit = (idx >> site) & 1
        phase = np.where(bit == 0, 1.0j, -1.0j)
        h[idx ^ (1 << site), idx] += phase
    else:
        raise ValueError(f"unknown Pauli {pauli}")
    return h


def build_hamiltonian(
    model: str,
    n: int,
    *,
    g: float = 1.0,
    h: float = 0.0,
    delta: float = 1.0,
    boundary: str = "open",
) -> np.ndarray:
    """Return the dense real Hamiltonian for one round-1 model."""
    n = check_n(n)
    model = model.lower()
    boundary = "open" if boundary.lower() in {"open", "obc"} else "periodic"
    out = _zeros(n)
    if model in {"one_site_product", "product_x", "independent_spins"}:
        for i in range(n):
            add_x(out, i, -float(g))
        return out
    if model in {"disconnected_dimers", "dimers"}:
        if n % 2:
            raise ValueError("disconnected dimers require even n")
        for i in range(0, n, 2):
            add_xx(out, i, i + 1, 0.25)
            add_yy(out, i, i + 1, 0.25)
            add_zz(out, i, i + 1, 0.25)
        return out
    if model in {"commuting_zz_ising", "zz", "classical_ising"}:
        for i, j in bonds(n, boundary):
            add_zz(out, i, j, -1.0)
        return out
    if model in {"transverse_field_ising", "tfim"}:
        for i, j in bonds(n, boundary):
            add_zz(out, i, j, -1.0)
        for i in range(n):
            add_x(out, i, -float(g))
        return out
    if model in {"mixed_field_ising", "mixed"}:
        for i, j in bonds(n, boundary):
            add_zz(out, i, j, -1.0)
        for i in range(n):
            add_x(out, i, -float(g))
            add_z(out, i, -float(h))
        return out
    if model in {"xxz", "xxz_symmetry_negative_control"}:
        for i, j in bonds(n, boundary):
            add_xx(out, i, j, 1.0)
            add_yy(out, i, j, 1.0)
            add_zz(out, i, j, float(delta))
        return out
    if model in {"ising_2d_classical", "classical_ising_2d"}:
        raise ValueError("build 2d classical Ising with ising_2d_hamiltonian")
    raise ValueError(f"unknown model {model}")


def ising_2d_edges(lx: int, ly: int, boundary: str = "open") -> list[tuple[int, int]]:
    """Row-major sites, x fastest or y? Site (x,y) = x + lx*y, x along the row."""
    if boundary.lower() not in {"open", "obc"}:
        raise ValueError("2d helper currently builds open rectangles; cylinders use transfer matrices")
    edges = []
    for y in range(ly):
        for x in range(lx):
            s = x + lx * y
            if x + 1 < lx:
                edges.append((s, s + 1))
            if y + 1 < ly:
                edges.append((s, s + lx))
    return edges


def ising_2d_hamiltonian(lx: int, ly: int) -> np.ndarray:
    n = lx * ly
    out = _zeros(n)
    for i, j in ising_2d_edges(lx, ly):
        add_zz(out, i, j, -1.0)
    return out


def classical_energies_zz(n: int, boundary: str = "open") -> np.ndarray:
    """Diagonal of H=-sum ZZ. Useful for large classical chains without a dense matrix."""
    n = check_n(n)
    idx = np.arange(1 << n)
    energy = np.zeros(idx.shape, dtype=np.float64)
    for i, j in bonds(n, boundary):
        zi = 1 - 2 * ((idx >> i) & 1)
        zj = 1 - 2 * ((idx >> j) & 1)
        energy += -zi * zj
    return energy


def xxz_sector_indices(n: int, n_down: int) -> np.ndarray:
    idx = np.arange(1 << n)
    return idx[np.bitwise_count(idx) == n_down]


def xxz_sector_hamiltonian(n: int, n_down: int, delta: float = 1.0, boundary: str = "open") -> tuple[np.ndarray, np.ndarray]:
    """XXZ restricted to a fixed magnetization sector. Returns (H, basis)."""
    basis = xxz_sector_indices(n, n_down)
    lookup = -np.ones(1 << n, dtype=np.int64)
    lookup[basis] = np.arange(basis.size)
    dim = basis.size
    h = np.zeros((dim, dim), dtype=np.float64)
    idx = basis
    for i, j in bonds(n, boundary):
        zi = 1 - 2 * ((idx >> i) & 1)
        zj = 1 - 2 * ((idx >> j) & 1)
        h[np.arange(dim), np.arange(dim)] += float(delta) * zi * zj
        # XX+YY exchanges opposite bits and has matrix element 2.
        bi = (idx >> i) & 1
        bj = (idx >> j) & 1
        hop = bi != bj
        dest = idx ^ (1 << i) ^ (1 << j)
        src = np.flatnonzero(hop)
        dst = lookup[dest[src]]
        # XX contributes +1 and YY contributes +1 on these hops, total +2.
        h[dst, src] += 2.0
    return h, basis
