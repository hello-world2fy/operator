"""Spherical multi-start minimization of fixed-basis Operator Walshness.

The objective is quadratic, not linear, in each Bloch vector, so this is a
constrained optimization on (S^2)^n rather than the pure-state coordinate
update. Every returned minimum is an upper bound on the intrinsic value,
except at α=0 where `intrinsic_alpha0` is exact.
"""

from __future__ import annotations

import numpy as np
import torch

from .walsh import (
    fixed_x_directions,
    intrinsic_alpha0,
    walshness_from_directions,
)


def structured_starts(n: int, alpha0_axes: np.ndarray | None, neighbor: np.ndarray | None, restarts: int, rng: np.random.Generator) -> np.ndarray:
    starts = [
        fixed_x_directions(n),
        _axis(n, 2, 1.0),
        _axis(n, 1, 1.0),
        _staggered(n, 2),
        _staggered(n, 0),
    ]
    if alpha0_axes is not None:
        starts.append(np.asarray(alpha0_axes, dtype=np.float64))
    if neighbor is not None:
        starts.append(np.asarray(neighbor, dtype=np.float64))
    while len(starts) < restarts:
        draw = rng.normal(size=(n, 3))
        draw /= np.linalg.norm(draw, axis=1, keepdims=True)
        starts.append(draw)
    packed = np.stack(starts[:restarts], axis=0)
    return packed / np.linalg.norm(packed, axis=-1, keepdims=True)


def _axis(n: int, component: int, sign: float) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.float64)
    out[:, component] = sign
    return out


def _staggered(n: int, component: int) -> np.ndarray:
    out = np.zeros((n, 3), dtype=np.float64)
    for site in range(n):
        out[site, component] = 1.0 if site % 2 == 0 else -1.0
    return out


def _local_factors(directions: torch.Tensor, alpha: float) -> torch.Tensor:
    a = 0.5 * (1.0 + np.exp(alpha))
    b = 0.5 * (np.exp(alpha) - 1.0)
    nx, ny, nz = directions.unbind(-1)
    out = torch.empty(*nx.shape, 2, 2, dtype=torch.complex128, device=directions.device)
    out[..., 0, 0] = a - b * nz
    out[..., 1, 1] = a + b * nz
    out[..., 0, 1] = -b * nx + 1j * b * ny
    out[..., 1, 0] = -b * nx - 1j * b * ny
    return out


def _apply_ket(operator: torch.Tensor, local: torch.Tensor, site: int, n: int) -> torch.Tensor:
    del n
    batch, size, _ = operator.shape
    stride = 1 << site
    high = size // (2 * stride)
    tensor = operator.reshape(batch, high, 2, stride, size)
    # local is (batch, new, old); site 0 is the least significant bit.
    tensor = torch.einsum("bno,bholc->bhnlc", local, tensor)
    return tensor.reshape(batch, size, size)


def _apply_bra(operator: torch.Tensor, local: torch.Tensor, site: int, n: int) -> torch.Tensor:
    del n
    batch, size, _ = operator.shape
    stride = 1 << site
    high = size // (2 * stride)
    tensor = operator.reshape(batch, size, high, 2, stride)
    tensor = torch.einsum("bon,brhol->brhnl", local, tensor)
    return tensor.reshape(batch, size, size)


def batched_moment(operator: torch.Tensor, directions: torch.Tensor, alpha: float) -> torch.Tensor:
    """Moment Tr(O† A O A)/Tr(O†O) for a batch of Bloch configurations."""
    n = directions.shape[1]
    factors = _local_factors(directions, alpha)
    transformed = operator.unsqueeze(0).expand(directions.shape[0], -1, -1).clone()
    for site in range(n):
        local = factors[:, site]
        transformed = _apply_ket(transformed, local, site, n)
        transformed = _apply_bra(transformed, local, site, n)
    numerator = torch.einsum("ij,bij->b", operator.conj(), transformed).real
    denominator = torch.einsum("ij,ij->", operator.conj(), operator).real
    return numerator / denominator


def riemannian_gradient(operator: torch.Tensor, directions: np.ndarray, alpha: float, device: torch.device) -> np.ndarray:
    vector = torch.tensor(directions, dtype=torch.float64, device=device)
    vector = vector / vector.norm(dim=-1, keepdim=True)
    vector.requires_grad_(True)
    moment = batched_moment(operator, vector.unsqueeze(0), alpha)
    torch.log(moment).backward()
    grad = vector.grad
    radial = (grad * vector).sum(dim=-1, keepdim=True)
    tangent = grad - radial * vector
    return tangent.detach().cpu().numpy()


def _adam_chunk_size(operator: torch.Tensor, n_starts: int) -> int:
    """Keep the autograd tape of A(n) O A(n) inside one 80 GB GPU."""
    n = int(round(np.log2(operator.shape[0])))
    # The site contraction is a strided 2×2 einsum, so the tape holds one
    # matrix per site on each side. n>=12 keeps a smaller batch so four
    # devices can run together.
    copies = 8 if n >= 12 else (2 * n + 4)
    bytes_per_restart = operator.numel() * operator.element_size() * copies
    budget = 16 * (1 << 30)
    return max(1, min(n_starts, int(budget // max(bytes_per_restart, 1))))


def _adam_batch(
    operator: torch.Tensor,
    starts: np.ndarray,
    alpha: float,
    steps: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    chunk = _adam_chunk_size(operator, starts.shape[0])
    pieces = []
    for begin in range(0, starts.shape[0], chunk):
        block = starts[begin : begin + chunk]
        raw = torch.tensor(block, dtype=torch.float64, device=device)
        raw = torch.nn.Parameter(raw.clone())
        optimizer = torch.optim.Adam([raw], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            directions = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-15)
            moment = batched_moment(operator, directions, alpha)
            torch.log(moment).sum().backward()
            optimizer.step()
        with torch.no_grad():
            directions = raw / raw.norm(dim=-1, keepdim=True)
        pieces.append(directions.detach().cpu().numpy())
        del raw, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(pieces, axis=0)


def _lbfgs_one(
    operator: torch.Tensor,
    start: np.ndarray,
    alpha: float,
    device: torch.device,
    max_iter: int,
) -> np.ndarray:
    raw = torch.nn.Parameter(torch.tensor(start, dtype=torch.float64, device=device))
    optimizer = torch.optim.LBFGS(
        [raw],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        directions = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        moment = batched_moment(operator, directions.unsqueeze(0), alpha)
        loss = torch.log(moment).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        directions = raw / raw.norm(dim=-1, keepdim=True)
    return directions.detach().cpu().numpy()


def optimize_walshness(
    operator: np.ndarray,
    alpha: float,
    *,
    restarts: int = 24,
    seed: int = 0,
    neighbor: np.ndarray | None = None,
    steps: int = 250,
    lr: float = 0.08,
    polish: int = 40,
    device: str | torch.device = "cpu",
    gradient_tol: float = 1e-7,
) -> dict[str, np.ndarray | float | bool]:
    """Minimize W_α over local Bloch directions.

    The stored minimum is recomputed by the NumPy product formula. It is an
    upper bound unless ``certified_global`` is true.
    """
    alpha = float(alpha)
    operator = np.asarray(operator)
    n = int(np.log2(operator.shape[0]))
    alpha0_value, alpha0_axes = intrinsic_alpha0(operator)
    if alpha == 0.0:
        return {
            "W_upper": alpha0_value,
            "certified_global": True,
            "axes_best": alpha0_axes,
            "W_restarts": np.array([alpha0_value]),
            "axes_restarts": alpha0_axes[None, :, :],
            "gradient_norm": np.array([0.0]),
            "moment_disagreement": 0.0,
        }
    rng = np.random.default_rng(seed)
    starts = structured_starts(n, alpha0_axes, neighbor, restarts, rng)
    torch_device = torch.device(device)
    op = torch.tensor(operator, dtype=torch.complex128, device=torch_device)
    refined = _adam_batch(op, starts, alpha, steps=steps, lr=lr, device=torch_device)
    polished = []
    # Polish every restart when the system is small; otherwise polish the
    # structured starts and the current best Adam candidates.
    order = list(range(min(7, refined.shape[0])))
    adam_values = []
    for axes in refined:
        adam_values.append(walshness_from_directions(operator, axes, alpha))
    adam_values = np.asarray(adam_values)
    best_order = list(np.argsort(adam_values)[: min(4, refined.shape[0])])
    polish_ids = []
    for item in order + best_order:
        if item not in polish_ids:
            polish_ids.append(item)
    if n <= 8:
        polish_ids = list(range(refined.shape[0]))
    polished = np.array(refined, copy=True)
    for index in polish_ids:
        polished[index] = _lbfgs_one(op, refined[index], alpha, torch_device, polish)
    values = np.array([walshness_from_directions(operator, axes, alpha) for axes in polished])
    best = int(np.argmin(values))
    gradients = riemannian_gradient(op, polished[best], alpha, torch_device)
    # Cross-check the torch moment at the best point against NumPy.
    with torch.no_grad():
        torch_moment = float(
            batched_moment(
                op,
                torch.tensor(polished[best], dtype=torch.float64, device=torch_device).unsqueeze(0),
                alpha,
            ).cpu()
        )
    numpy_moment = float(np.exp(alpha * values[best]))
    return {
        "W_upper": float(values[best]),
        "certified_global": False,
        "axes_best": polished[best],
        "W_restarts": values,
        "axes_restarts": polished,
        "gradient_norm": np.linalg.norm(gradients, axis=-1),
        "gradient_within_tol": bool(np.linalg.norm(gradients) <= gradient_tol * max(n, 1) ** 0.5),
        "moment_disagreement": abs(torch_moment - numpy_moment) / max(numpy_moment, 1e-300),
        "W_alpha0": alpha0_value,
        "fixed_W": float(walshness_from_directions(operator, fixed_x_directions(n), alpha)),
    }
