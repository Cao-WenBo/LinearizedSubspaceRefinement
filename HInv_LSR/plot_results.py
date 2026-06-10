"""Plot the main HeatInv LSR reproduction figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hinv_core import noise_tag, parse_float_list


STYLE = {
    "0": {"color": "#1F4E79", "linestyle": "-", "linewidth": 2.3},
    "0.001": {"color": "#3E78A8", "linestyle": "-", "linewidth": 2.2},
    "0.002": {"color": "#6FA6BF", "linestyle": "-", "linewidth": 2.2},
    "0.005": {"color": "#9CB8B3", "linestyle": "-", "linewidth": 2.2},
    "0.01": {"color": "#C7A76C", "linestyle": "-", "linewidth": 2.2},
    "0.02": {"color": "#C77945", "linestyle": "-", "linewidth": 2.2},
    "0.05": {"color": "#9B5143", "linestyle": "-", "linewidth": 2.2},
    "0.1": {"color": "#5A5A5A", "linestyle": "-", "linewidth": 2.2},
}


def noise_label(noise: float) -> str:
    if noise == 0:
        return "0"
    if noise < 0.01:
        return f"{noise:.0e}"
    return f"{noise:g}"


def load_scan(lsr_dir: Path, noise: float, seed: int) -> pd.DataFrame:
    path = lsr_dir / f"hinv_lsr_noise_{noise_tag(noise)}_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).sort_values("rank")


def selected_row(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["val_noisy_u_mse_lsr"].astype(float).idxmin()]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Times New Roman"],
            "font.size": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_rank_curves(lsr_dir: Path, noises, seed: int, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 4.65), sharex=True)
    for noise in noises:
        df = load_scan(lsr_dir, noise, seed)
        sel = selected_row(df)
        style = STYLE.get(f"{noise:g}", STYLE["0.1"])
        label = noise_label(noise)
        axes[0].plot(df["rank"], df["eval_u_error_lsr"], label=label, **style)
        axes[1].plot(df["rank"], df["eval_a_error_lsr"], label=label, **style)
        axes[0].scatter(
            sel["rank"],
            sel["eval_u_error_lsr"],
            s=34,
            marker="o",
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.4,
            zorder=5,
        )
        axes[1].scatter(
            sel["rank"],
            sel["eval_a_error_lsr"],
            s=34,
            marker="o",
            facecolors="white",
            edgecolors=style["color"],
            linewidths=1.4,
            zorder=5,
        )
    axes[0].set_ylabel(r"$u$ error")
    axes[1].set_ylabel(r"$a$ error")
    axes[1].set_xlabel("Rank")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(which="major", axis="y", linestyle="--", color="#D0D0D0", linewidth=0.6, alpha=0.55)
        ax.tick_params(labelsize=10)
    axes[0].legend(title="Noise level", ncol=2, loc="lower left", fontsize=8, title_fontsize=9, frameon=False)
    fig.tight_layout(h_pad=0.15)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".tiff"]:
        fig.savefig(fig_dir / f"hinv_rank_curves{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(lsr_dir: Path, noises, seeds, fig_dir: Path) -> None:
    rows = []
    for noise in noises:
        for seed in seeds:
            df = load_scan(lsr_dir, noise, seed)
            sel = selected_row(df)
            base = df[df["rank"].astype(int) == 0].iloc[0]
            rows.append(
                {
                    "noise": noise,
                    "seed": seed,
                    "rank": int(sel["rank"]),
                    "pinn_a": float(base["eval_a_error"]),
                    "lsr_a": float(sel["eval_a_error_lsr"]),
                    "pinn_u": float(base["eval_u_error"]),
                    "lsr_u": float(sel["eval_u_error_lsr"]),
                }
            )
    selected = pd.DataFrame(rows)
    selected.to_csv(fig_dir / "selected_rank_summary.csv", index=False)

    grouped = selected.groupby("noise", sort=True)
    labels = [noise_label(float(x)) for x in grouped.mean(numeric_only=True).index]
    x = np.arange(len(labels), dtype=float)
    mean = grouped.mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(4.2, 4.8), sharex=True)
    axes[0].plot(x - 0.08, mean["pinn_a"], color="#1F4E79", marker="o", linewidth=1.9, markersize=4.2, label="PINN")
    axes[0].plot(x + 0.08, mean["lsr_a"], color="#C77945", marker="s", linewidth=1.9, markersize=4.2, label="LSR")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$a$ error")
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].plot(x, mean["rank"], color="#5A5A5A", marker="o", linewidth=1.9, markersize=4.2)
    axes[1].set_ylabel("Selected rank")
    axes[1].set_xlabel("Noise level")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylim(bottom=0)

    for ax in axes:
        ax.grid(which="major", axis="y", linestyle=":", color="#D0D0D0", linewidth=0.7, alpha=0.65)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    for suffix in [".png", ".pdf", ".tiff"]:
        fig.savefig(fig_dir / f"hinv_selected_summary{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot HeatInv LSR reproduction results.")
    p.add_argument("--lsr-dir", default="outputs/lsr")
    p.add_argument("--fig-dir", default="outputs/figures")
    p.add_argument("--noise-list", default="0 0.001 0.01 0.1")
    p.add_argument("--seed-list", default="0")
    p.add_argument("--rank-curve-seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    noises = parse_float_list(args.noise_list)
    seeds = [int(x) for x in args.seed_list.replace(",", " ").split()]
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_rank_curves(Path(args.lsr_dir), noises, args.rank_curve_seed, fig_dir)
    plot_summary(Path(args.lsr_dir), noises, seeds, fig_dir)
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()

