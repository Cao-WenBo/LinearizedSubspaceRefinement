"""Train LRA PINN baselines for the HeatInv inverse problem."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from hinv_core import (
    DEFAULT_DATA_POINTS,
    FNN,
    adapt_lra_weights,
    evaluate_fields,
    evaluate_validation,
    loss_terms,
    make_problem_data,
    noise_tag,
    parse_float_list,
    parse_hidden_layers,
    set_seed,
)


def save_checkpoint(path: Path, model: FNN, optimizer: torch.optim.Optimizer, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def run_one(args, noise: float, seed: int, summary_path: Path) -> dict:
    device = torch.device(args.device)
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    set_seed(seed)

    run_dir = Path(args.output_dir) / f"noise_{noise_tag(noise)}_seed_{seed}"
    if run_dir.exists() and (run_dir / "best_val.pt").exists() and not args.overwrite:
        print(f"Skip completed run: {run_dir}")
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
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

    layers = [3] + parse_hidden_layers(args.hidden_layers) + [2]
    model = FNN(layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.ones(3, device=device, dtype=dtype)
    best_val = float("inf")
    best_metrics = {}
    history_rows = []
    start = time.time()

    for step in range(args.iterations + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, data)
            if args.use_lra:
                weights = adapt_lra_weights(model, terms, weights, alpha=args.lra_alpha).to(device=device, dtype=dtype)
            total = sum(weights[i] * terms[i] for i in range(3))
            total.backward()
            optimizer.step()

        if step % args.val_every == 0 or step == args.iterations:
            terms_eval = loss_terms(model, data)
            val_metrics = evaluate_validation(model, data)
            is_best = val_metrics["val_noisy_u_mse"] < best_val
            eval_metrics = evaluate_fields(model, data.eval_x) if (is_best or not args.eval_on_best_only) else {}
            if is_best:
                best_val = val_metrics["val_noisy_u_mse"]
                best_metrics = {**val_metrics, **eval_metrics}
                save_checkpoint(
                    run_dir / "best_val.pt",
                    model,
                    optimizer,
                    {
                        "noise": noise,
                        "seed": seed,
                        "step": step,
                        "layers": layers,
                        "loss_weights": [float(x) for x in weights.detach().cpu()],
                        "best_val_noisy_u_mse": best_val,
                        "metrics": best_metrics,
                    },
                )

            row = {
                "step": step,
                "noise": noise,
                "seed": seed,
                "pde_loss": float(terms_eval[0].detach().cpu()),
                "data_loss": float(terms_eval[1].detach().cpu()),
                "bc_loss": float(terms_eval[2].detach().cpu()),
                "pde_weight": float(weights[0].detach().cpu()),
                "data_weight": float(weights[1].detach().cpu()),
                "bc_weight": float(weights[2].detach().cpu()),
                "val_noisy_u_mse": val_metrics["val_noisy_u_mse"],
                "val_clean_u_mse": val_metrics["val_clean_u_mse"],
                "val_clean_u_error": val_metrics["val_clean_u_error"],
                "eval_u_error": eval_metrics.get("eval_u_error", float("nan")),
                "eval_a_error": eval_metrics.get("eval_a_error", float("nan")),
                "best_val_noisy_u_mse": best_val,
            }
            history_rows.append(row)
            print(
                f"noise={noise:g} seed={seed} step={step:05d} "
                f"val={val_metrics['val_noisy_u_mse']:.3e} "
                f"best={best_val:.3e} w={weights.detach().cpu().numpy()}"
            )

    final_metrics = {**evaluate_validation(model, data), **evaluate_fields(model, data.eval_x)}
    save_checkpoint(
        run_dir / "final.pt",
        model,
        optimizer,
        {
            "noise": noise,
            "seed": seed,
            "step": args.iterations,
            "layers": layers,
            "loss_weights": [float(x) for x in weights.detach().cpu()],
            "metrics": final_metrics,
        },
    )

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "noise": noise,
        "seed": seed,
        "run_dir": str(run_dir),
        "iterations": args.iterations,
        "train_points": int(data.train_x.shape[0]),
        "val_points": int(data.val_x.shape[0]),
        "best_val_noisy_u_mse": best_val,
        "best_eval_u_error": best_metrics.get("eval_u_error", float("nan")),
        "best_eval_a_error": best_metrics.get("eval_a_error", float("nan")),
        "final_eval_u_error": final_metrics["eval_u_error"],
        "final_eval_a_error": final_metrics["eval_a_error"],
        "time_sec": time.time() - start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    append_header = not summary_path.exists()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if append_header:
            writer.writeheader()
        writer.writerow(summary)

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train HeatInv LRA baselines.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    p.add_argument("--data-points", default=str(DEFAULT_DATA_POINTS))
    p.add_argument("--output-dir", default="outputs/baselines")
    p.add_argument("--summary-csv", default="outputs/baseline_summary.csv")
    p.add_argument("--noise-list", default="0 0.001 0.01 0.1")
    p.add_argument("--seed-list", default="0")
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-layers", default="100*5")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--eval-on-best-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-lra", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lra-alpha", type=float, default=0.1)
    p.add_argument("--n-domain", type=int, default=4096)
    p.add_argument("--n-boundary-pde", type=int, default=1024)
    p.add_argument("--n-initial", type=int, default=1024)
    p.add_argument("--n-bc", type=int, default=2048)
    p.add_argument("--eval-points", type=int, default=8192)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    noises = parse_float_list(args.noise_list)
    seeds = [int(x) for x in args.seed_list.replace(",", " ").split()]
    summary_path = Path(args.summary_csv)
    for seed in seeds:
        for noise in noises:
            run_one(args, noise, seed, summary_path)


if __name__ == "__main__":
    main()

