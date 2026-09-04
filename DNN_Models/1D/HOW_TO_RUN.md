# How to run the 1-D model

This folder reproduces the 1-D (constant mutation-rate) study end to end.
One script does everything.

## What it does, briefly

A neural network is fit to predict the mean **and the variance** of the
summary statistic `d̄` as a function of `log10(p)`, the log mutation
probability — a heteroscedastic nonlinear regression, trained by maximum
likelihood and then conformally calibrated so its predictive intervals have
valid coverage. That fitted model is substituted for the expensive exact
simulator inside an ABC-MCMC (Metropolis–Hastings) sampler, exactly the way
the paper's Gaussian-process surrogate is used. The pipeline then compares
five estimators — MOM, MLE, exact ABC-MCMC, GPS-ABC (the paper's GP) and
DNN-ABC (this network) — on estimation accuracy, credible-interval width,
coverage, and compute time, over a grid of true mutation rates `p` and
culture counts `J`.

The script runs four steps in order:

| Step | What happens | Time |
|---|---|---|
| 1 | Train + conformally calibrate the surrogate | ~30 seconds |
| 2 | Run all five estimators on many simulated data sets → Tables 1–3 | **hours** |
| 3 | Monte Carlo standard errors — which differences are real vs. noise | seconds |
| 4 | Regenerate all figures | seconds |

**Step 2 is genuinely slow, and that is the point.** It includes the exact
ABC-MCMC baseline, which re-simulates the branching process cell by cell at
every MCMC iteration — roughly 340 seconds per 100 iterations at `p=1e-4,
J=100`, versus 0.135 seconds for either surrogate. That ~2500× gap is the
paper's headline result, so reproducing it means actually paying that cost
once. Use `--quick` first if you just want to confirm the pipeline runs.

## Running it — Mac / Linux

Open Terminal, `cd` into this folder, then:

```bash
./run_all.sh            # full run, paper settings
./run_all.sh --quick    # 1-2 minute smoke test first, if you prefer
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
python network\train.py
python abc\run_experiments.py --reps 40 --nmcmc 600 --burnin 250 --ns 6 --p-grid 1e-4 1e-3 1e-2 --J-grid 10 50 100
python abc\mcse.py
python figures\make_figures.py
```

Requires Python 3.9 or newer. No GPU needed — everything runs on CPU.

## What to expect

**On the first run**, the script creates a `.venv/` folder and installs
packages (numpy, pandas, scipy, matplotlib, scikit-learn, torch) — a few
minutes of download the first time only. Subsequent runs skip this.

**While running**, you'll see labelled progress: `[0/4]` through `[4/4]`,
training output showing early-stopping and the conformal calibration factor,
then a progress counter with an ETA during the long comparison step
(`[120/360] done  elapsed 4.2m  eta 8.4m`), then timing results per cell.

**Runtime**: `--quick` takes 1–2 minutes. The full run takes **several hours**
— it is dominated by the exact-simulator baseline at the small-`p`, large-`J`
cells (see the note above), and replicates already run in parallel across all
your CPU cores. Nothing is wrong if step 2 sits at a low task count for a long
while; the slowest single cell alone is tens of minutes of genuine simulation.

**When it finishes**, everything lands in `results/`:

- `results/tables/TABLES.md` — Tables 1, 2 and 3, formatted for reading
- `results/tables/mcse.md` — Monte Carlo standard errors, i.e. which
  table differences are statistically resolved and which are ties
- `results/figures/*.png` — all result figures
- `results/model/` — the trained surrogate and its fit metrics

`--quick` produces the same files with far fewer replicates, so the numbers
will be noisier than the reported ones — use it to confirm the pipeline runs,
not to read results off.

## Don't want to run anything?

`results/` is already populated with the committed outputs of a full run, so
`results/tables/TABLES.md`, `results/tables/mcse.md` and the figures can be
read directly with no Python environment at all.

## Folder map

```
1D/
├── run_all.sh       runs everything (this is the entry point)
├── HOW_TO_RUN.md    this file
├── requirements.txt Python packages
├── data/            ground-truth data from the exact simulator (included)
├── network/         the neural network: architecture + training
├── abc/             the simulator, the classical estimators, and the ABC-MCMC sampler
├── figures/         figure generation
└── results/         all outputs (already populated)
```

Each file opens with a comment block explaining what it does and which part
of the method it implements.
