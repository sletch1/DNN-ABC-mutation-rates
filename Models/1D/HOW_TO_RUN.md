# How to run the 1-D model

Constant mutation rate — the paper's Study 1.

---

## 1. What you need

Python 3.9 or newer. Nothing else: no GPU, no compiler, no server, no data to
download. The ground-truth data is already in `data/`.

## 2. Run it

### The easy way: `run_all.py`

`run_all.py` runs the whole pipeline. There are **no command-line arguments to
type**: it asks in the console which run you want, and you answer with 1 or 2.

```
Which run do you want?

  [1] Quick  - a smoke test, roughly 3 minutes. Confirms the pipeline works
               end to end. Its numbers are noisy: do not read results off it.

  [2] Full   - the paper's settings: 40 replicates, 3 mutation rates, 3 culture
               counts. Several hours (see section 6). You very likely do not
               need this -- results/ already holds a full run.

Enter 1 or 2 [1]:
```

**From an IDE (Spyder, VS Code, PyCharm, IDLE):** open `run_all.py` and press
**Run**. Answer the question in the console pane, and that is the whole job.
In Spyder the answer goes in the IPython console at the bottom right — click
into it, type `1`, press Enter.

**From a terminal:** `cd` into this folder and run `python run_all.py`
(`python3` on Mac/Linux). To skip the question — a server job, say — pass
`--quick` or `--full`.

The pipeline runs with whatever interpreter your IDE is using, so the packages
in `requirements.txt` have to be installed there. If any are missing the script
says which, and offers to install them for you; answer `y` once and it handles
it.

Each step prints its output to the console as it runs, and the run stops with
the error visible if a step fails. Press 1 for the first run — read section 6
before choosing 2.

### Setting up the packages yourself (optional)

If you would rather not let the script install anything, do it once by hand.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script ("running scripts is disabled"),
either use `.venv\Scripts\activate.bat` instead, or allow it once with:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

Mac / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the steps one at a time

`run_all.py` is only a wrapper around these commands, so you can run them
yourself instead. The paper's settings:

```bash
python network/train.py --seed 0
python abc/run_experiments.py --reps 40 --nmcmc 600 --burnin 250 --ns 6 --p-grid 1e-4 1e-3 1e-2 --J-grid 10 50 100
python abc/mcse.py
python figures/make_figures.py
```

**To check it runs first without waiting hours**, use these smaller settings for
the second command, then read §6 before doing the full version:

```bash
python abc/run_experiments.py --reps 2 --nmcmc 120 --burnin 40 --ns 6 --p-grid 1e-2 --J-grid 10 --workers 2
python abc/mcse.py
python figures/make_figures.py --quick
```

On Windows use backslashes in the script paths (`python network\train.py ...`).

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

## 4. What you'll see while it runs

`run_all.py` asks which run you want, then prints four labelled steps. Real
output, abbreviated:

```
Interpreter: C:\Users\you\anaconda3\python.exe
Folder:      C:\...\Models\1D

Which run do you want?

  [1] Quick  - a smoke test, roughly 3 minutes. ...
  [2] Full   - the paper's settings: 40 replicates, ...

Enter 1 or 2 [1]: 1
=== QUICK MODE: pipeline check only, NOT paper-scale results ===

--- [1/4] Training the surrogate (fast, well under a minute) ---
train n=505  val n=303  test n=202
Early stopping at epoch 132 (best val NLL=-2.0804)
conformal sd_scale = 1.0500
[test] n= 202  MSE(log)=0.00435  MAE(log)=0.05238  MSE(d_bar)=4.952e-05  95%cover=0.950

--- [2/4] Estimator comparison: Tables 1, 2 and 3 (this is the long one) ---
=== Phase A: accuracy (Table 1 & 2) ===
accuracy: 360 tasks (3p x 3J x 40 reps) on 30 workers
  [10/360] done  elapsed 11.0m  eta 385.4m
  ...
=== Phase B: timing (Table 3) ===

--- [3/4] Monte Carlo standard errors ---
--- [4/4] Figures ---
Done in 412.6 min. Everything written to results/:
```

Each step also echoes the exact command it is running, so if one fails you can
re-run that single command on its own to see the error again.

**Step 2 is where the time goes**, and it prints a running count with an ETA so
you can see it is progressing. The ETA is honest but jumps around early on,
because the cells differ enormously in cost — small mutation rates with many
cultures are the slow ones.

Nothing else prints for long stretches. That is normal, not a hang: a single
slow cell can be tens of minutes of genuine simulation. If you want to confirm
it is alive, the task counter is the thing to watch.

## 5. What comes out

Everything lands in `results/`:

| File | What it is |
|---|---|
| `results/tables/TABLES.md` | Tables 1–3, formatted for reading |
| `results/tables/mcse.md` | Monte Carlo standard errors — which differences are real and which are noise |
| `results/figures/*.png` | all figures |
| `results/model/` | the trained surrogate and its fit metrics |

`results/` is **already populated** with a committed full run, so you can read
all of the above without running anything.

## 6. Why the full run takes hours

The pipeline compares five estimators — MOM, MLE, exact ABC-MCMC, GPS-ABC (the
paper's Gaussian process) and DNN-ABC (this network). The exact ABC-MCMC
baseline re-simulates the branching process cell by cell at every MCMC
iteration: roughly **340 seconds per 100 iterations** at `p=1e-4, J=100`,
against **0.135 seconds** for either surrogate.

That ~2500× gap *is* the headline result, so reproducing it means paying the
cost once. Replicates run in parallel across all your cores, but there are 360
of them (3 mutation rates x 3 culture counts x 40 replicates).

**Concretely: on a 32-core Linux server using 30 workers, the full run takes
about 7 hours.** On a laptop with 4-8 cores, expect several times that. If step
2 sits at a low task count for a long time, nothing is wrong -- a single slow
cell is tens of minutes of genuine simulation.

**You very likely do not need to run this.** `results/` already contains the
committed output of a full run, and §5 lists what is in it. Run the full
pipeline only if you specifically want to reproduce the timings on your own
hardware; otherwise `--quick` confirms the code works and the committed results
are there to read.

`--quick` shrinks the expensive parts (2 replicates, one cheap `p`/`J` cell, a
short posterior chain), so it confirms the pipeline runs but its numbers are
noisy. Do not read results off a `--quick` run.

## 7. What the model does

A neural network is fit to predict the mean **and the variance** of the summary
statistic `d̄ = meanᵢ √(Xᵢ/Zᵢ)` as a function of `log10(p)` — a heteroscedastic
regression, trained by maximum likelihood, then conformally calibrated so its
predictive intervals have valid coverage. That fitted model replaces the
expensive simulator inside an ABC-MCMC sampler, exactly where the paper puts its
Gaussian-process surrogate.

## 8. Folder map

```
1D/
├── run_all.py       the entry point — press Run in an IDE, or `python run_all.py`
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
