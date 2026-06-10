# HeatInv LSR Reproduction

This folder contains a self-contained reproduction package for the HeatInv
inverse problem used in the LSR paper.

The benchmark is adapted from the PINNacle `HeatInv` inverse problem.  The goal
is to recover the spatial coefficient field `a(x,y)` in a transient heat
equation from noisy temperature observations.

## Problem

The governing equation is

```text
u_t - div(a grad u) = f(x,y,t),     (x,y) in [-1,1]^2, t in [0,1].
```

The manufactured solution is

```text
u(x,y,t) = exp(-t) sin(pi x) sin(pi y),
a(x,y)   = 2 + sin(pi x) sin(pi y).
```

The file `data/heatinv_points.dat` contains the 2500 observation locations used
by the original PINNacle benchmark.  No external PINNacle checkout is required
to run the scripts in this folder.

## Files

- `hinv_core.py`: equation definition, exact solution, FNN, LRA utilities, and sampling.
- `train_lra_baseline.py`: trains LRA PINN baselines and saves best-validation checkpoints.
- `run_lsr_scan.py`: loads the saved baselines and performs LSR rank scans.
- `plot_results.py`: plots error-vs-rank curves and selected-rank summaries.
- `run_demo.py`: compact end-to-end runner.
- `data/heatinv_points.dat`: observation locations.

## Installation

```bash
pip install -r requirements.txt
```

The LSR script uses `torch.func`, so PyTorch 2.0 or newer is recommended.

## Quick demo

From this folder:

```bash
python run_demo.py --device cuda:0 --noise-list "0 0.001 0.01 0.1" --seed-list "0" --iterations 20000 --rank-max 1000 --rank-step 100
```

This trains one LRA baseline for each selected noise level, applies LSR, and
saves figures in `outputs/figures`.

For a faster smoke test:

```bash
python run_demo.py --device cuda:0 --noise-list "0" --seed-list "0" --iterations 10 --rank-max 20 --rank-step 20
```

The smoke test only checks that the pipeline runs; it is not expected to match
paper accuracy.

## Main-effect reproduction

The manuscript experiment uses multiple noise levels and three random seeds.
The commands below reproduce the main effect: low-noise LSR gives a large
coefficient-field improvement, while the validation-selected rank decreases as
the observation noise increases.

```bash
python train_lra_baseline.py --device cuda:0 --noise-list "0 0.001 0.002 0.005 0.01 0.02 0.05 0.1" --seed-list "0 1 2" --iterations 20000 --val-every 100

python run_lsr_scan.py --device cuda:0 --noise-list "0 0.001 0.002 0.005 0.01 0.02 0.05 0.1" --seed-list "0 1 2" --rank-max 3000 --rank-step 50 --jvp-chunk-size 4

python plot_results.py --noise-list "0 0.001 0.002 0.005 0.01 0.02 0.05 0.1" --seed-list "0 1 2"
```

The key outputs are:

- `outputs/figures/hinv_rank_curves.png`: `u` and `a` error vs rank.
- `outputs/figures/hinv_selected_summary.png`: baseline PINN vs validation-selected LSR and selected rank vs noise.
- `outputs/figures/selected_rank_summary.csv`: selected ranks and errors.

## Notes

- The LRA baseline uses the same three loss components as the original HeatInv
  benchmark: PDE residual, noisy `u` observations, and boundary values for `a`.
- LSR freezes the final LRA loss weights saved in the best-validation checkpoint.
- Rank zero corresponds to the unmodified baseline prediction.  Thus validation
  can reject LSR when the observation noise is too large.
- This folder is a compact PyTorch reproduction, not a verbatim DeepXDE/PINNacle
  training launcher.  The equation, data split, LRA weighting rule, validation
  selection rule, and LSR residual construction match the manuscript workflow,
  while the default residual sample counts are kept moderate so the package is
  easy to run.  They can be increased with `--n-domain`, `--n-boundary-pde`,
  `--n-initial`, and `--n-bc`.
