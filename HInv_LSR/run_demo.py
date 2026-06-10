"""Run a compact end-to-end HeatInv LSR reproduction."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Compact HeatInv LSR demo runner.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--noise-list", default="0 0.001 0.01 0.1")
    p.add_argument("--seed-list", default="0")
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--rank-max", type=int, default=1000)
    p.add_argument("--rank-step", type=int, default=100)
    p.add_argument("--jvp-chunk-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    run(
        [
            py,
            "train_lra_baseline.py",
            "--device",
            args.device,
            "--noise-list",
            args.noise_list,
            "--seed-list",
            args.seed_list,
            "--iterations",
            str(args.iterations),
        ]
    )
    run(
        [
            py,
            "run_lsr_scan.py",
            "--device",
            args.device,
            "--noise-list",
            args.noise_list,
            "--seed-list",
            args.seed_list,
            "--rank-max",
            str(args.rank_max),
            "--rank-step",
            str(args.rank_step),
            "--jvp-chunk-size",
            str(args.jvp_chunk_size),
        ]
    )
    run(
        [
            py,
            "plot_results.py",
            "--noise-list",
            args.noise_list,
            "--seed-list",
            args.seed_list,
        ]
    )


if __name__ == "__main__":
    main()

