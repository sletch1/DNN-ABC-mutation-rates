# How to run the 4-D model

Two-stage mutation rate, parameters `(log10 p1, log10 p2, tau, log10 delta)` — the paper's Study 2.
Run everything from inside this folder (`4D/`); all paths below are relative to it.

> From the repository root, `make study2` runs every step below (and `make paper` runs the whole paper, Studies I and II and the Ricker benchmark), skipping steps whose output exists. This file documents the individual commands.

---

## 1. What you need

- **Python 3.9+** with the packages in `../requirements.txt` (numpy, pandas, scipy, matplotlib, scikit-learn,
  threadpoolctl, torch, sbi).
- **No GPU needed.** Everything runs on CPU. More cores make it faster (the multi-core steps use a process pool).
- **MATLAB** only for the GPS-ABC comparison (step 6). Everything else is Python.
- **bash** for the `server/*.sh` drivers (Mac/Linux; on Windows use WSL or Git Bash, or run the `python` commands
  inside them by hand). The drivers call `nproc` and `/proc/loadavg`, which are Linux-only; on a Mac, run the
  `python` commands directly (as written in sections 3 to 5) or pass explicit `--cores`.

Setup once:

```bash
cd 4D
python3 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r ../requirements.txt
```

## 2. What is already included (so you can skip steps)

| Already in the folder | Lets you skip |
|---|---|
| `data/slow_data_4D.csv` (training data) and `data/obs_4d_case{1,2}.npz` (observed datasets) | step 1 |
| `results/surrogate_4d.pt` (trained, selected surrogate) and the other `surrogate_*.pt` | steps 2 and 3 |
| `results/arch_search_4D_*.csv/json`, `results/tables/*.csv`, `figures/*.pdf` | the finished outputs of 2 and 7 |

Not included (large and regenerable): the DNN-ABC / NPE chains (`results/dnnabc*.npz`, `results/npe*.npz`),
the GPS-ABC chains (`results/gps/`), and the MATLAB observed-data export (`data/mat/`). Steps 4 to 7 recreate them.

**Fastest useful path:** do the check in section 3, then step 4 (DNN-ABC with the shipped surrogate).

## 3. Check the install (about 10 seconds)

```bash
python tests/test_sim.py
```

Expected last line: `ok: mean mutants exact/fast = 2.31 (1-stage), 2.52 (2-stage); LHS strata complete`
(the numbers vary a little with the random seed).

## 4. The pipeline

Settings follow the original paper: `t_p = 20`, `J = 100`, `p in [1e-11, 1e-7]`, `tau in [0.1, 19.9]`,
`delta in [0.5, 2]`. Two observed cases (`case 1`, `case 2`), 100 observed datasets each.

### Step 1 — Generate data (about 1 minute on 32 cores)

```bash
python datagen/gen_training.py --model 4d --n 10000 --rep 5 --out data/slow_data_4D.csv
for c in 1 2; do
  python datagen/gen_observed.py --model 4d --case $c --nsim 100 --out data/obs_4d_case$c.npz
done
```

Quick test of the generator only: `python datagen/gen_training.py --model 4d --n 20 --rep 2 --cores 2 --out /tmp/t.csv`
(takes a couple of seconds). Overwriting `data/` changes every downstream result, so write test output elsewhere.

### Step 2 — Train the surrogate: architecture search, then select and train the final model

The 12 architectures x 5 seeds search, then the selection and final fit (these are the commands the
`README.md` runs the same two commands):

```bash
python model/arch_search.py --data data/slow_data_4D.csv --seeds 5 --out results/arch_search_4D
python model/select_and_report.py --data data/slow_data_4D.csv --search results/arch_search_4D --out results
```

Output: `results/surrogate_4d.pt`, `results/surrogate_4d_metrics.json`, and the architecture-selection report.
This is the slow Python step (hours on a laptop; it parallelises across cores). Use `--max-cores N` to cap it.

### Step 3 — DNN-ABC, NPE and simulation-based calibration (SBC)

```bash
bash server/run_study2.sh 4d            # DNN-ABC on both cases, NPE, SBC
bash server/finish_study2.sh 4d         # NPE for case 2 (reuses the case-1 NPE), then SBC
```

`run_study2.sh` takes an optional core cap: `bash server/run_study2.sh 4d 8`. To use a specific interpreter:
`PY=/path/to/venv/bin/python bash server/run_study2.sh 4d`.

Or run one DNN-ABC job by hand (100 chains, 35,000 iterations each: 10,000 adaptation + 5,000 burn-in, both
discarded, then 20,000 kept):

```bash
python inference/dnn_abc.py --surrogate results/surrogate_4d.pt --obs data/obs_4d_case1.npz \
       --out results/dnnabc_4d_case1.npz --step-scale 0.05 --cores 8
```

Smoke test (finishes in seconds, acceptance will be about 0 because adaptation has not finished — that is
expected, it only proves the pipeline runs): add `--iters 600 --n 10` and write to a scratch `--out`.

Output: `results/dnnabc_4d_case{1,2}.npz`, `results/npe_4d_case{1,2}.{npz,pkl}`, `results/sbc_4d.{npz,json}`.

### Step 4 — Controls

```bash
bash server/run_matched.sh              # DNN-ABC trained on only 1,500 design points (matched to GPS-ABC)
bash server/run_corr.sh                 # DNN-ABC-rho: surrogate that models the two features jointly
python model/gp_baseline.py --data data/slow_data_4D.csv --out results/gp_baseline_4D.json
```

Optional argument to both shell scripts: architecture name and cores, e.g. `bash server/run_matched.sh res_128x3_gelu 6`.

### Step 5 — Validate the simulator against the original authors' (optional)

```bash
python tests/validate_vs_published.py
```

Needs `../Original_Paper_Code/training/`, which is kept local and **not** in this zip, so this only works on the
original machine. Its result is already stored in `results/validation_vs_published.json`.

### Step 6 — GPS-ABC with MATLAB (optional, about hours per replicate)

The authors' GPS-ABC code (`../Original_Paper_Code/`), run on our observed data. MATLAB with the Parallel
Computing Toolbox is required.

```bash
# export the observed data for MATLAB
python datagen/export_mat.py data/obs_4d_case1.npz data/mat/obs_4d_case1
python datagen/export_mat.py data/obs_4d_case2.npz data/mat/obs_4d_case2
# build the GP initialisation once (from gps_abc/)
(cd gps_abc && matlab -batch "make_init(12)")
# one chain: case, dataset index (seed), workers, chain length
(cd gps_abc && matlab -batch "run_rep(1, 1, 2, 20000)")
```

Several chains at once (first K datasets of each case):

```bash
bash server/run_gps_4d.sh 6 20000 12      # K=6 datasets, N=20000 iterations, 12 chains in parallel
```

Chains are checkpointed every 250 iterations and **resume automatically** if you start the same
case/seed again. Copy `gps_abc/out/ckpt_*.mat` into `results/gps/` before analysing (the analysis reads that
folder). `server/gps_queue.sh jobs.txt` runs a queue of `CASE SEED N [TAG]` lines with a concurrency limit; an example
is `server/jobs_gps_extra.txt`.

### Step 7 — Tables and figures

```bash
python analysis/study2_analysis.py --model 4d                   # recovery / coverage / fidelity tables -> results/tables/
python analysis/study2_analysis.py --model 4d --gps-horizon 2000   # same, GPS-ABC compared at a common chain length
python analysis/make_tables_tex.py                              # LaTeX tables for the manuscript
python analysis/make_figures.py --model 4d                      # PDFs in figures/
```

`study2_analysis.py` compares whichever methods have results: it needs the `results/dnnabc*.npz` chains from
steps 3 and 4, and `results/gps/ckpt_*.mat` for the GPS-ABC column. If a method's files are missing, that method
is skipped.

## 5. Which files come from which step

| Step | Produces |
|---|---|
| 1 | `data/slow_data_4D.csv`, `data/obs_4d_case{1,2}.npz` |
| 2 | `results/arch_search_4D_{raw.csv,meta.json}`, `results/surrogate_4d.pt` |
| 3 | `results/dnnabc_4d_case*.npz`, `results/npe_4d_case*`, `results/sbc_4d.*` |
| 4 | `results/dnnabc1500_*`, `results/dnnabcrho*`, `results/surrogate_4d_n1500.pt`, `results/surrogate_4d_rho*.pt`, `results/gp_baseline_4D.json` |
| 6 | `gps_abc/out/gps4d_case*_*.mat`, `gps_abc/out/ckpt_*.mat` |
| 7 | `results/tables/*.csv`, `figures/*.pdf` |

## 6. Troubleshooting

- **`nproc: command not found` or `/proc/loadavg` errors (Mac):** the shell drivers are written for a Linux server.
  Run the `python ...` command inside the script by hand, with `--cores N`.
- **`ModuleNotFoundError`:** the virtual environment is not active, or `pip install -r ../requirements.txt` did not finish.
- **Out of memory in GPS-ABC:** `gps_abc/patched/mut_bMBP_rev.m` is a memory-safe copy of the authors' function and is
  already on the path in `run_rep.m`; lower the number of parallel chains.
- **Different numbers from the manuscript:** the random seeds are fixed in the scripts, but results depend on the
  number of cores and library versions for the multi-process steps, so small differences are normal.
