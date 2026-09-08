# How to run the 3-D model

This folder reproduces the 3-D (two-stage mutation-rate) study end to end.
One script does everything.

## What it does, briefly

The mutation rate is no longer constant: it jumps from `p1` to `p2` at an
unknown time `τ`, and all three parameters are estimated **jointly**. A neural
network is fit to predict the mean **and the variance** of the summary
statistic `d̄` as a function of `(log10 p1, log10 p2, τ)` — a heteroscedastic
nonlinear regression, trained by maximum likelihood and then conformally
calibrated so its predictive intervals have valid coverage. That fitted model
is substituted for the expensive exact simulator inside an ABC-MCMC sampler,
the same way the paper's Gaussian-process surrogate is used. The pipeline then
compares the two surrogates — GPS-ABC (the paper's GP) and DNN-ABC (this
network) — on recovery of each parameter, credible-interval width, coverage,
and compute time.

**The headline result is negative, and is reported as such.** The two
surrogates are statistically tied in eight of nine parameter-by-truth
comparisons. What binds here is not surrogate error but the model's own
identifiability: `p1` and `τ` are weakly determined by the data, so a better
surrogate cannot help.

The script runs four steps in order:

| Step | What happens | Time |
|---|---|---|
| 1 | Train + conformally calibrate the surrogate | ~30 seconds |
| 2 | Run both ABC methods on many simulated data sets → recovery table | ~2 minutes |
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

**Training is deterministic** at `--seed 0`: it reproduces the committed
`results/model/surrogate_metrics.json` exactly, so you can check your
environment is sound by confirming the test row reads
`mse_mean = 1.658e-3 (1.19x its 2-rep floor)`.

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
├── matlab/          reference MATLAB for the two-stage simulator
├── figures/         figure generation
├── run_on_server.sh regenerates the ground-truth data (not needed to run the study)
└── results/         all outputs (already populated)
```

Each file opens with a comment block explaining what it does and which part
of the method it implements.
