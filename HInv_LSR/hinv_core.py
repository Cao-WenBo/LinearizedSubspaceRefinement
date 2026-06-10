"""Core utilities for the PINNacle HeatInv inverse problem.

This module is intentionally self-contained.  It implements the HeatInv
manufactured solution, a small FNN, LRA-style loss-weight adaptation, and the
sampling routines used by both the baseline trainer and LSR scanner.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_POINTS = THIS_DIR / "data" / "heatinv_points.dat"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def noise_tag(noise: float) -> str:
    if noise == 0:
        return "0"
    return f"{noise:g}".replace("-", "m").replace(".", "p")


def parse_hidden_layers(spec: str) -> List[int]:
    if "*" in spec:
        width, depth = spec.split("*", 1)
        return [int(width)] * int(depth)
    return [int(x) for x in spec.split("_") if x]


class FNN(nn.Module):
    def __init__(self, layer_sizes: Sequence[int]):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears[:-1]:
            x = torch.tanh(layer(x))
        return self.linears[-1](x)


def u_ref_np(xyt: np.ndarray) -> np.ndarray:
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def a_ref_np(xyt: np.ndarray) -> np.ndarray:
    x, y = xyt[:, 0:1], xyt[:, 1:2]
    return 2.0 + np.sin(np.pi * x) * np.sin(np.pi * y)


def u_ref(xyt: torch.Tensor) -> torch.Tensor:
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    return torch.exp(-t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def a_ref(xyt: torch.Tensor) -> torch.Tensor:
    x, y = xyt[:, 0:1], xyt[:, 1:2]
    return 2.0 + torch.sin(math.pi * x) * torch.sin(math.pi * y)


def f_src(xyt: torch.Tensor) -> torch.Tensor:
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    s, c, p = torch.sin, torch.cos, math.pi
    return torch.exp(-t) * (
        (4.0 * p**2 - 1.0) * s(p * x) * s(p * y)
        + p**2
        * (
            2.0 * s(p * x) ** 2 * s(p * y) ** 2
            - c(p * x) ** 2 * s(p * y) ** 2
            - s(p * x) ** 2 * c(p * y) ** 2
        )
    )


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-30)).detach().cpu().item()


def rel_l2_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), 1e-30))


def to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def sample_spacetime(n: int, rng: np.random.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pts = rng.random((n, 3), dtype=np.float32)
    pts[:, 0:2] = 2.0 * pts[:, 0:2] - 1.0
    return to_tensor(pts, device, dtype)


def sample_spatial_boundary(n: int, rng: np.random.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pts = rng.random((n, 3), dtype=np.float32)
    pts[:, 0:2] = 2.0 * pts[:, 0:2] - 1.0
    side_dim = rng.integers(0, 2, size=n)
    side_val = rng.integers(0, 2, size=n).astype(np.float32)
    pts[np.arange(n), side_dim] = 2.0 * side_val - 1.0
    return to_tensor(pts, device, dtype)


def sample_initial(n: int, rng: np.random.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pts = rng.random((n, 3), dtype=np.float32)
    pts[:, 0:2] = 2.0 * pts[:, 0:2] - 1.0
    pts[:, 2] = 0.0
    return to_tensor(pts, device, dtype)


@dataclass
class HInvData:
    train_x: torch.Tensor
    train_u: torch.Tensor
    val_x: torch.Tensor
    val_u_noisy: torch.Tensor
    val_u_clean: torch.Tensor
    pde_x: torch.Tensor
    bc_x: torch.Tensor
    eval_x: torch.Tensor


def make_problem_data(
    *,
    data_points: Path,
    noise: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    val_ratio: float = 0.2,
    n_domain: int = 4096,
    n_boundary_pde: int = 1024,
    n_initial: int = 1024,
    n_bc: int = 2048,
    n_eval: int = 8192,
) -> HInvData:
    rng = np.random.default_rng(seed + 9173)
    data_pts = np.loadtxt(data_points).astype("float32")
    clean_u = u_ref_np(data_pts).astype("float32")
    noisy_u = clean_u + rng.normal(0.0, noise, size=clean_u.shape).astype("float32")

    indices = rng.permutation(data_pts.shape[0])
    n_val = int(round(data_pts.shape[0] * val_ratio))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    sample_rng = np.random.default_rng(seed + 12345)
    domain_x = sample_spacetime(n_domain, sample_rng, device, dtype)
    boundary_pde_x = sample_spatial_boundary(n_boundary_pde, sample_rng, device, dtype)
    initial_x = sample_initial(n_initial, sample_rng, device, dtype)
    pde_x = torch.cat([domain_x, boundary_pde_x, initial_x], dim=0)
    bc_x = sample_spatial_boundary(n_bc, sample_rng, device, dtype)
    eval_x = sample_spacetime(n_eval, sample_rng, device, dtype)

    return HInvData(
        train_x=to_tensor(data_pts[train_idx], device, dtype),
        train_u=to_tensor(noisy_u[train_idx], device, dtype),
        val_x=to_tensor(data_pts[val_idx], device, dtype),
        val_u_noisy=to_tensor(noisy_u[val_idx], device, dtype),
        val_u_clean=to_tensor(clean_u[val_idx], device, dtype),
        pde_x=pde_x,
        bc_x=bc_x,
        eval_x=eval_x,
    )


def pde_residual(model: nn.Module, xyt: torch.Tensor) -> torch.Tensor:
    xyt = xyt.detach().clone().requires_grad_(True)
    ua = model(xyt)
    u, a = ua[:, 0:1], ua[:, 1:2]
    grad_u = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    au_x, au_y = a * u_x, a * u_y
    d_au_x = torch.autograd.grad(au_x, xyt, torch.ones_like(au_x), create_graph=True)[0][:, 0:1]
    d_au_y = torch.autograd.grad(au_y, xyt, torch.ones_like(au_y), create_graph=True)[0][:, 1:2]
    return u_t - d_au_x - d_au_y - f_src(xyt)


def loss_terms(model: nn.Module, data: HInvData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pde_loss = torch.mean(pde_residual(model, data.pde_x) ** 2)
    pred_data = model(data.train_x)[:, 0:1]
    data_loss = torch.mean((pred_data - data.train_u) ** 2)
    pred_bc_a = model(data.bc_x)[:, 1:2]
    bc_loss = torch.mean((pred_bc_a - a_ref(data.bc_x)) ** 2)
    return pde_loss, data_loss, bc_loss


def adapt_lra_weights(
    model: nn.Module,
    losses: Sequence[torch.Tensor],
    weights: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    pde_grads = torch.autograd.grad(losses[0], params, retain_graph=True, allow_unused=True)
    max_pde = max(
        [g.detach().abs().max().item() for g in pde_grads if g is not None] + [1e-12]
    )
    new_weights = weights.detach().clone()
    for i in range(1, len(losses)):
        grads = torch.autograd.grad(losses[i], params, retain_graph=True, allow_unused=True)
        total_abs = 0.0
        total_count = 0
        for grad in grads:
            if grad is None:
                continue
            total_abs += torch.sum(torch.abs(grad.detach())).item()
            total_count += grad.numel()
        mean_grad = total_abs / max(total_count, 1)
        target = max_pde / max(mean_grad * float(new_weights[i]), 1e-12)
        new_weights[i] = (1.0 - alpha) * new_weights[i] + alpha * target
    return new_weights


@torch.no_grad()
def evaluate_fields(model: nn.Module, xyt: torch.Tensor) -> Dict[str, float]:
    pred = model(xyt)
    target_u, target_a = u_ref(xyt), a_ref(xyt)
    return {
        "eval_u_mse": torch.mean((pred[:, 0:1] - target_u) ** 2).item(),
        "eval_a_mse": torch.mean((pred[:, 1:2] - target_a) ** 2).item(),
        "eval_u_error": rel_l2(pred[:, 0:1], target_u),
        "eval_a_error": rel_l2(pred[:, 1:2], target_a),
    }


@torch.no_grad()
def evaluate_validation(model: nn.Module, data: HInvData) -> Dict[str, float]:
    pred_u = model(data.val_x)[:, 0:1]
    return {
        "val_noisy_u_mse": torch.mean((pred_u - data.val_u_noisy) ** 2).item(),
        "val_clean_u_mse": torch.mean((pred_u - data.val_u_clean) ** 2).item(),
        "val_clean_u_error": rel_l2(pred_u, data.val_u_clean),
    }
