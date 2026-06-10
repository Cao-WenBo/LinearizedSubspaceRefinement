"""Linearized Subspace Refinement rank scan for trained HeatInv baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.func import functional_call, hessian, jacrev, jvp, vjp, vmap

from hinv_core import (
    DEFAULT_DATA_POINTS,
    FNN,
    a_ref,
    f_src,
    make_problem_data,
    noise_tag,
    parse_float_list,
    rel_l2,
    set_seed,
    u_ref,
)


class ParamVectorizer:
    def __init__(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]):
        self.meta = []
        pieces = []
        for name, param in named_parameters:
            base = param.detach().clone().cpu()
            self.meta.append((name, base))
            pieces.append(base.reshape(-1))
        self.theta0_cpu = torch.cat(pieces)

    @property
    def numel(self) -> int:
        return int(self.theta0_cpu.numel())

    def theta0(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.theta0_cpu.to(device=device, dtype=dtype)

    def unpack(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        ptr = 0
        for name, base_cpu in self.meta:
            k = base_cpu.numel()
            out[name] = theta[ptr : ptr + k].reshape(base_cpu.shape)
            ptr += k
        return out


class HeatInvLSR:
    def __init__(
        self,
        model: FNN,
        vectorizer: ParamVectorizer,
        data,
        theta0: torch.Tensor,
        loss_weights: Sequence[float],
        jvp_chunk_size: int = 4,
    ):
        self.model = model
        self.vectorizer = vectorizer
        self.data = data
        self.theta0 = theta0
        self.device = theta0.device
        self.dtype = theta0.dtype
        self.jvp_chunk_size = int(jvp_chunk_size)
        self.base_params = {name: param.detach() for name, param in model.named_parameters()}
        self.pde_weight, self.data_weight, self.bc_weight = [float(x) for x in loss_weights]
        self.pde_scale = math.sqrt(self.pde_weight / max(int(data.pde_x.shape[0]), 1))
        self.data_scale = math.sqrt(self.data_weight / max(int(data.train_x.shape[0]), 1))
        self.bc_scale = math.sqrt(self.bc_weight / max(int(data.bc_x.shape[0]), 1))

    def params_for_theta(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        params = dict(self.base_params)
        params.update(self.vectorizer.unpack(theta))
        return params

    def predict(self, theta: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.params_for_theta(theta), (xyt,))

    def ns_residual(self, theta: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        def one_residual(coord):
            def out_fn(x):
                return self.predict(theta, x.unsqueeze(0)).squeeze(0)

            y = out_fn(coord)
            jac = jacrev(out_fn)(coord)
            hess_u = hessian(lambda z: out_fn(z)[0])(coord)
            u_x, u_y, u_t = jac[0, 0], jac[0, 1], jac[0, 2]
            a_x, a_y = jac[1, 0], jac[1, 1]
            a = y[1]
            div_au = a_x * u_x + a * hess_u[0, 0] + a_y * u_y + a * hess_u[1, 1]
            return u_t - div_au - f_src(coord.unsqueeze(0)).squeeze()

        return vmap(one_residual)(xyt).reshape(-1, 1)

    def system(self, theta: torch.Tensor) -> torch.Tensor:
        pieces = []
        pde = self.ns_residual(theta, self.data.pde_x)
        pieces.append(self.pde_scale * pde)
        pred_data = self.predict(theta, self.data.train_x)[:, 0:1]
        pieces.append(self.data_scale * (pred_data - self.data.train_u))
        pred_bc = self.predict(theta, self.data.bc_x)[:, 1:2]
        pieces.append(self.bc_scale * (pred_bc - a_ref(self.data.bc_x)))
        return torch.cat(pieces, dim=0)

    def jv(self, vectors: torch.Tensor) -> torch.Tensor:
        cols = vectors.T

        def lin(delta):
            _, tangent = jvp(self.system, (self.theta0,), (delta,))
            return tangent.flatten()

        return torch.cat(
            [vmap(lin)(chunk) for chunk in cols.split(self.jvp_chunk_size)], dim=0
        ).T.detach()

    def jtv(self, gradients: torch.Tensor) -> torch.Tensor:
        rows = gradients.T
        _, vjp_fn = vjp(lambda th: self.system(th).flatten(), self.theta0)
        return torch.cat(
            [vmap(vjp_fn)(chunk)[0] for chunk in rows.split(self.jvp_chunk_size)], dim=0
        ).T.detach()

    def prepare_basis(self, max_rank: int, oversample: int):
        k = int(max_rank + oversample)
        start = time.time()
        omega = torch.randn(self.vectorizer.numel, k, device=self.device, dtype=self.dtype)
        jo = self.jv(omega)
        if jo.shape[0] < k:
            raise ValueError(
                f"rank+oversample={k} exceeds residual rows={jo.shape[0]}. "
                "Reduce --rank-max or increase residual sample counts."
            )
        q, _ = torch.linalg.qr(jo, mode="reduced")
        jtq = self.jtv(q.detach())
        _, singular, vh = torch.linalg.svd(jtq.T, full_matrices=False)
        basis = vh.T[:, :max_rank].detach()
        amat = self.jv(basis) if max_rank > 0 else torch.empty(jo.shape[0], 0, device=self.device, dtype=self.dtype)
        bvec = -self.system(self.theta0).detach()
        return basis, amat, bvec, float(time.time() - start), singular[:max_rank].detach().cpu()

    def solve_rank(self, rank: int, basis_full: torch.Tensor, amat_full: torch.Tensor, bvec: torch.Tensor):
        if rank == 0:
            return torch.zeros_like(self.theta0), 0.0
        basis = basis_full[:, :rank]
        amat = amat_full[:, :rank]
        coeff = torch.linalg.lstsq(amat, bvec).solution
        delta = (basis @ coeff).flatten().detach()
        lin_sse = torch.sum((bvec + amat @ coeff) ** 2).detach().cpu().item()
        return delta, lin_sse

    def evaluate(self, delta: torch.Tensor) -> Dict[str, float]:
        r0, tangent = jvp(self.system, (self.theta0,), (delta,))
        train_loss = torch.sum(r0**2).detach().cpu().item()
        train_loss_lsr = torch.sum((r0 + tangent) ** 2).detach().cpu().item()

        pred0, pred_tan = jvp(lambda th: self.predict(th, self.data.eval_x), (self.theta0,), (delta,))
        pred1 = pred0 + pred_tan
        target_u, target_a = u_ref(self.data.eval_x), a_ref(self.data.eval_x)

        v0, vtan = jvp(lambda th: self.predict(th, self.data.val_x)[:, 0:1], (self.theta0,), (delta,))
        v1 = v0 + vtan
        return {
            "train_loss": train_loss,
            "train_loss_lsr": train_loss_lsr,
            "val_noisy_u_mse": torch.mean((v0 - self.data.val_u_noisy) ** 2).detach().cpu().item(),
            "val_noisy_u_mse_lsr": torch.mean((v1 - self.data.val_u_noisy) ** 2).detach().cpu().item(),
            "val_clean_u_error": rel_l2(v0.detach().cpu(), self.data.val_u_clean.detach().cpu()),
            "val_clean_u_error_lsr": rel_l2(v1.detach().cpu(), self.data.val_u_clean.detach().cpu()),
            "eval_u_error": rel_l2(pred0[:, 0:1].detach(), target_u.detach()),
            "eval_u_error_lsr": rel_l2(pred1[:, 0:1].detach(), target_u.detach()),
            "eval_a_error": rel_l2(pred0[:, 1:2].detach(), target_a.detach()),
            "eval_a_error_lsr": rel_l2(pred1[:, 1:2].detach(), target_a.detach()),
        }


def write_rows(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def scan_one(args, checkpoint: Path, results_path: Path) -> List[dict]:
    device = torch.device(args.device)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    obj = torch.load(checkpoint, map_location="cpu")
    metadata = obj.get("metadata", {})
    noise = float(metadata.get("noise", args.noise))
    seed = int(metadata.get("seed", args.seed))
    layers = metadata.get("layers", [3, 100, 100, 100, 100, 100, 2])
    weights = metadata.get("loss_weights", [1.0, 1.0, 1.0])

    set_seed(seed)
    model = FNN(layers).to(device=device, dtype=dtype)
    model.load_state_dict(obj["model_state_dict"])
    model.eval()
    data = make_problem_data(
        data_points=Path(args.data_points),
        noise=noise,
        seed=seed,
        device=device,
        dtype=dtype,
        val_ratio=args.val_ratio,
        n_domain=args.n_domain,
        n_boundary_pde=args.n_boundary_pde,
        n_initial=args.n_initial,
        n_bc=args.n_bc,
        n_eval=args.eval_points,
    )
    vectorizer = ParamVectorizer(model.named_parameters())
    theta0 = vectorizer.theta0(device, dtype)
    lsr = HeatInvLSR(model, vectorizer, data, theta0, weights, jvp_chunk_size=args.jvp_chunk_size)

    ranks = list(range(args.rank_start, args.rank_max + 1, args.rank_step))
    max_rank = max(ranks)
    print(
        f"LSR checkpoint={checkpoint} noise={noise:g} seed={seed} "
        f"param_dim={vectorizer.numel:,} max_rank={max_rank} weights={weights}"
    )
    basis, amat, bvec, prep_sec, singular = lsr.prepare_basis(max_rank, args.oversample)

    rows = []
    for rank in ranks:
        start = time.time()
        delta, lin_sse = lsr.solve_rank(rank, basis, amat, bvec)
        metrics = lsr.evaluate(delta)
        row = {
            "rank": int(rank),
            "noise": noise,
            "seed": seed,
            "param_dim": vectorizer.numel,
            "pde_weight": float(weights[0]),
            "data_weight": float(weights[1]),
            "bc_weight": float(weights[2]),
            "prep_sec": prep_sec,
            "solve_eval_sec": time.time() - start,
            "linear_sse": lin_sse,
            **metrics,
        }
        rows.append(row)
        print(
            f"  rank={rank:4d} val={row['val_noisy_u_mse_lsr']:.3e} "
            f"u={row['eval_u_error_lsr']:.3e} a={row['eval_a_error_lsr']:.3e}"
        )
        write_rows(rows, results_path)

    best = min(rows, key=lambda r: r["val_noisy_u_mse_lsr"])
    print(
        f"Best rank={best['rank']} noise={noise:g} "
        f"u/a={best['eval_u_error_lsr']:.3e}/{best['eval_a_error_lsr']:.3e}"
    )
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Run HeatInv LSR rank scans.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    p.add_argument("--data-points", default=str(DEFAULT_DATA_POINTS))
    p.add_argument("--baseline-dir", default="outputs/baselines")
    p.add_argument("--output-dir", default="outputs/lsr")
    p.add_argument("--noise-list", default="0 0.001 0.01 0.1")
    p.add_argument("--seed-list", default="0")
    p.add_argument("--noise", type=float, default=0.0, help="Fallback if checkpoint metadata is missing.")
    p.add_argument("--seed", type=int, default=0, help="Fallback if checkpoint metadata is missing.")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--n-domain", type=int, default=4096)
    p.add_argument("--n-boundary-pde", type=int, default=1024)
    p.add_argument("--n-initial", type=int, default=1024)
    p.add_argument("--n-bc", type=int, default=2048)
    p.add_argument("--eval-points", type=int, default=8192)
    p.add_argument("--rank-start", type=int, default=0)
    p.add_argument("--rank-max", type=int, default=1000)
    p.add_argument("--rank-step", type=int, default=100)
    p.add_argument("--oversample", type=int, default=10)
    p.add_argument("--jvp-chunk-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    noises = parse_float_list(args.noise_list)
    seeds = [int(x) for x in args.seed_list.replace(",", " ").split()]
    for seed in seeds:
        for noise in noises:
            checkpoint = Path(args.baseline_dir) / f"noise_{noise_tag(noise)}_seed_{seed}" / "best_val.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing baseline checkpoint: {checkpoint}")
            results = Path(args.output_dir) / f"hinv_lsr_noise_{noise_tag(noise)}_seed_{seed}.csv"
            scan_one(args, checkpoint, results)


if __name__ == "__main__":
    main()

