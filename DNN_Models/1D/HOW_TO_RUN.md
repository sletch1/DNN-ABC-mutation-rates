# How to run the 1-D model

Constant mutation rate — the paper's Study 1.

---

## 1. What you need

Python 3.9 or newer. Nothing else: no GPU, no compiler, no server, no data to
download. The ground-truth data is already in `data/`.

## 2. Run it

### Windows (PowerShell) — recommended

Open PowerShell, `cd` into this folder, and run these once to set up:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script ("running scripts is disabled"),
either use `.venv\Scripts\activate.bat` instead, or allow it once with:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Then run the pipeline:

```powershell
python network\train.py --seed 0
python abc\run_experiments.py --reps 40 --nmcmc 600 --burnin 250 --ns 6 --p-grid 1e-4 1e-3 1e-2 --J-grid 10 50 100
python abc\mcse.py
python figures\make_figures.py
```

**To check it runs first without waiting hours**, use these smaller settings for
the second command, then read §5 before doing the full version:

```powershell
python abc\run_experiments.py --reps 2 --nmcmc 120 --burnin 40 --p-grid 1e-2 --J-grid 10 --workers 2
python abc\mcse.py
python figures\make_figures.py --quick
```

### Windows (Git Bash or WSL) — if you prefer the one-command version

`run_all.sh` needs a bash shell. Install
[Git for Windows](https://git-scm.com/download/win), right-click in this folder →
"Git Bash Here", then:

```bash
./run_all.sh --quick     # ~3 minutes — checks everything works
./run_all.sh             # the real run — several hours (see §5)
```

If you get `bad interpreter` or `\r: command not found`, Git checked the script
out with Windows line endings. Fix it with:
`git config core.autocrlf input` and re-clone, or run `dos2unix run_all.sh`.

### Mac / Linux

```bash
./run_all.sh --quick     # ~3 minutes — checks everything works
                         # (~5 the very first time: see note below)
./run_all.sh             # the real run — several hours (see §5)
```

If you get a permissions error: `chmod +x run_all.sh`, then retry.

**First run only:** the script creates its own `.venv/` and downloads the
packages, which adds roughly two minutes. Every run after that reuses it. Both
cold-start paths were tested end to end and complete cleanly.

## 3. Check it worked

Training is deterministic. Run just the training step:

```bash
python network/train.py --seed 0
```

You should see:

```
conformal sd_scale = 1.0500
[train] n= 505  MSE(log)=0.00425  MAE(log)=0.05176  MSE(d_bar)=2.713e-05  95%cover=0.958
[test] n= 202  MSE(log)=0.00435  MAE(log)=0.05238  MSE(d_bar)=4.952e-05  95%cover=0.950
```

If those match, your environment is fine. (The very last digits in
`results/model/surrogate_metrics.json` may differ — ordinary floating-point
variation between machines, not a problem.)

**Do this before starting the full run**, so you don't wait hours to find out
something was wrong.

## 4. What comes out

Everything lands in `results/`:

| File | What it is |
|---|---|
| `results/tables/TABLES.md` | Tables 1–3, formatted for reading |
| `results/tables/mcse.md` | Monte Carlo standard errors — which differences are real and which are noise |
| `results/figures/*.png` | all figures |
| `results/model/` | the trained surrogate and its fit metrics |

`results/` is **already populated** with a committed full run, so you can read
all of the above without running anything.

## 5. Why the full run takes hours

The pipeline compares five estimators — MOM, MLE, exact ABC-MCMC, GPS-ABC (the
paper's Gaussian process) and DNN-ABC (this network). The exact ABC-MCMC
baseline re-simulates the branching process cell by cell at every MCMC
iteration: roughly **340 seconds per 100 iterations** at `p=1e-4, J=100`,
against **0.135 seconds** for either surrogate.

That ~2500× gap *is* the headline result, so reproducing it means paying the
cost once. Replicates already run in parallel across all your cores. If step 2
sits at a low task count for a long time, nothing is wrong — a single slow cell
is tens of minutes of genuine simulation.

`--quick` shrinks the expensive parts (2 replicates, one cheap `p`/`J` cell, a
short posterior chain), so it confirms the pipeline runs but its numbers are
noisy. Do not read results off a `--quick` run.

## 6. What the model does

A neural network is fit to predict the mean **and the variance** of the summary
statistic `d̄ = meanᵢ √(Xᵢ/Zᵢ)` as a function of `log10(p)` — a heteroscedastic
regression, trained by maximum likelihood, then conformally calibrated so its
predictive intervals have valid coverage. That fitted model replaces the
expensive simulator inside an ABC-MCMC sampler, exactly where the paper puts its
Gaussian-process surrogate.

## 7. Folder map

```
1D/
├── run_all.sh       the entry point
├── HOW_TO_RUN.md    this file
├── requirements.txt Python packages
├── data/            ground-truth data (included)
├── network/         the network: model.py (architecture), train.py (fitting)
├── abc/             simulator, MOM/MLE estimators, surrogates, ABC-MCMC sampler
├── matlab/          your original MATLAB, byte-identical — everything in abc/
│                    and network/ is a port of these
├── figures/         figure generation
└── results/         all outputs (already populated)
```

Every file opens with a comment block explaining what it does and which part of
the method it implements.

**On the imports at the top of `network/train.py`:** `model`, `surrogates` and
`paths` are files in this folder — `network/model.py`, `abc/surrogates.py` and
`paths.py`. `train.py` adds its sibling folders to `sys.path` itself, so
`python network/train.py` works from any directory as long as you have the whole
`1D/` folder. They are not installable packages; there is nothing to `pip
install` for them.
