# How to run the 3-D model

This folder reproduces the 3-D (two-stage mutation-rate) study end to end.
One script does everything.

## Before you start

You need **Python 3.9 or newer** and nothing else — no GPU, no compiler, no
server access. The ground-truth data is already in `data/`, so there is nothing
to download or generate. A full run takes about **five minutes**.

If you only want to read the results, skip to "Don't want to run anything?"
below; `results/` is already populated.

## What it does, briefly

The mutation rate is no longer constant: it jumps from `p1` to `p2` at an
unknown time `τ`, and all three parameters are estimated **jointly** from a
single scalar summary statistic. Following Sec. 2.2 of the paper, that
statistic is the **fourth** root, `S = meanᵢ (Xᵢ/Zᵢ)^(1/4)` — the paper uses
`√(X/Z)` for the constant-rate study but switches to the fourth root for the
two-stage model. A neural network is fit to predict the mean **and the
variance** of `log10 S` as a function of `(log10 p1, log10 p2, τ)` — a
heteroscedastic nonlinear regression, trained by maximum likelihood and then
conformally calibrated so its predictive intervals have valid coverage. That
fitted model is substituted for the expensive exact simulator inside an
ABC-MCMC sampler, the same way the paper's Gaussian-process surrogate is used.

The pipeline compares **three** surrogates on recovery of each parameter,
credible-interval width, coverage and compute time:

| Method | What it is |
|---|---|
| `DNN-ABC` | this project's neural surrogate |
| `GPS-ABC` | a GP baseline, strengthened: anisotropic kernel, log-scaled inputs |
| `GPS-ABC-ref` | a GP built to match `matlab/demoGPS_fluc_exp2.m` exactly |

Both GPs are reported because the strengthened one is better than the reference
on every axis, so quoting only it would overstate what the published baseline
achieves.

**The headline result is negative, and is reported as such.** DNN-ABC and
GPS-ABC are statistically tied in **seven of nine** parameter-by-truth
comparisons; the other two favour GPS-ABC and none favours the network. What
binds here is not surrogate error but the model's own identifiability: `p1` and
`τ` are weakly determined by one scalar summary, so a better surrogate cannot
help.

The script runs four steps in order:

| Step | What happens | Time |
|---|---|---|
| 1 | Train + conformally calibrate the surrogate | ~30 seconds |
| 2 | Run all three ABC methods on many simulated data sets → recovery table | ~3 minutes |
| 3 | Monte Carlo standard errors — which differences are real vs. noise | seconds |
| 4 | Regenerate all figures | seconds |

**Unlike the 1-D study, this one is fast.** The exact-simulator ABC baseline —
which re-simulates the branching process cell by cell at every MCMC iteration
and costs hours — is **off by default**, because this study compares the two
surrogates to each other rather than to the exact sampler. That matches the
reported results (`results/logs/experiment_config.json` records
`"with_sim": false`). Pass `--with-sim` to turn it on and expect hours.

## Running it — Mac / Linux

Open Terminal, `cd` into this folder, then:

```bash
./run_all.sh             # full run, reported settings (a few minutes)
./run_all.sh --quick     # ~1 minute smoke test first, if you prefer
./run_all.sh --with-sim  # also run the exact-simulator baseline (hours)
```

If you get a permissions error, run `chmod +x run_all.sh` once, then retry.

## Running it — Windows

The script needs a bash shell. Windows ships one with Git for Windows:

1. Install [Git for Windows](https://git-scm.com/download/win) if you don't
   have it (this also installs "Git Bash").
2. Right-click inside this folder → **"Git Bash Here"**.
3. Run the same command:

```bash
./run_all.sh            # or: ./run_all.sh --quick
```

WSL (Windows Subsystem for Linux) works identically if you already use it.

**If you'd rather not use bash at all**, the four steps are just Python
commands — run these from PowerShell inside this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python network\train.py --data data\slow_data_3D.csv --seed 0
python tests\validate_simulator.py --quick
python abc\run_experiments.py --reps 16 --nmcmc 3000 --burnin 1000 --ns 4 --J-grid 100 --no-sim
python abc\mcse.py
python figures\make_figures.py
```

Requires Python 3.9 or newer. No GPU needed — everything runs on CPU.

## What to expect

**On the first run**, the script creates a `.venv/` folder and installs
packages (numpy, pandas, scipy, matplotlib, scikit-learn, torch) — a few
minutes of download the first time only. Subsequent runs skip this.

**While running**, you'll see labelled progress: `[0/4]` through `[4/4]`,
training output showing the irreducible noise floor and the conformal
calibration factor, then a progress counter with an ETA during the comparison
step.

**Check your environment with this.** Training is deterministic at `--seed 0`
and reproduces the committed run to the precision shown below (the last digits
of `surrogate_metrics.json` can differ, which is ordinary floating-point
variation across BLAS builds and nothing to worry about). Run just step 1:

```bash
python network/train.py --seed 0
```

and confirm the last two lines read

```
conformal sd_scale = 1.0185
[test ] n= 4000  mse_mean=3.628e-04 (1.05x its 2-rep floor)  mse_obs=7.062e-04  95%cover=0.955
```

If those match, your environment is sound and everything downstream will
reproduce. If they do not, stop there — nothing later will match either.

**A second, independent check** (about 40 seconds) verifies the simulator
itself rather than the fit:

```bash
python tests/validate_simulator.py --quick
```

It confirms the two-stage simulator reduces exactly to the constant-rate model
in both limiting cases, and re-derives from the data which mutation-time
convention the ground truth was generated under. All checks should print PASS.

**When it finishes**, everything lands in `results/`:

- `results/tables/TABLES.md` — parameter recovery, formatted for reading
- `results/tables/mcse.md` — Monte Carlo standard errors, i.e. which
  differences are statistically resolved and which are ties
- `results/figures/*.png` — all result figures
- `results/model/` — the trained surrogate and its fit metrics

`--quick` produces the same files with far fewer replicates, so the numbers
will be noisier than the reported ones — use it to confirm the pipeline runs,
not to read results off.

## Don't want to run anything?

`results/` is already populated with the committed outputs of a full run, so
`results/tables/TABLES.md`, `results/tables/mcse.md` and the figures can be
read directly with no Python environment at all.

## Regenerating the ground-truth data

`data/slow_data_3D.csv` (2,000 Latin-hypercube design points × 10 replicates,
35 MB) is **included**, so you do not need to regenerate it. It was produced by
the exact cell-by-cell R simulator on a compute server, which takes hours:

```bash
bash run_on_server.sh          # full run (2000 x 10), needs SSH access to stat86
bash run_on_server.sh 20 2     # smoke test (20 x 2)
```

The R source is in `../../RCode/` (`funMBP.R`, `genSlowData_3D.R`), and the
reference MATLAB it was ported from is in `matlab/`.

## Folder map

```
3D/
├── run_all.sh       runs everything (this is the entry point)
├── HOW_TO_RUN.md    this file
├── requirements.txt Python packages
├── data/            ground-truth data from the exact simulator (included)
├── network/         the neural network: architecture, training, architecture search
├── abc/             the two-stage simulator, the surrogates, and the ABC-MCMC sampler
├── matlab/          reference MATLAB: the two-stage simulator (mut2stage_bMBP.m)
│                 and the Study 2 GP driver (demoGPS_fluc_exp2.m)
├── tests/           simulator degeneracy + ground-truth provenance checks
├── figures/         figure generation
├── run_on_server.sh regenerates the ground-truth data (not needed to run the study)
└── results/         all outputs (already populated)
```

Each file opens with a comment block explaining what it does and which part
of the method it implements.
