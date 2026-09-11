# How to run the 3-D model

Two-stage mutation rate `(p1, p2, τ)` — the paper's Study 2.

---

## 1. What you need

Python 3.9 or newer. Nothing else: no GPU, no compiler, no server, no data to
download. The ground-truth data is already in `data/`.

**Unlike the 1-D study, this one is fast — under 5 minutes end to end**
(measured: 3m40s on a 14-core laptop). The
expensive exact-simulator baseline is off by default, because this study
compares surrogates to each other rather than to the exact sampler
(`results/logs/experiment_config.json` records `"with_sim": false`).

## 2. Run it

### The easy way: `run_all.py`

`run_all.py` runs the whole pipeline. There are **no command-line arguments to
type**: it asks in the console which run you want, and you answer with 1, 2 or 3.

```
Which run do you want?

  [1] Quick  - a smoke test, roughly 1.5 minutes. Confirms the pipeline works
               end to end. Its numbers are noisy: do not read results off it.

  [2] Full   - the reported settings: 16 replicates, a few minutes. This is
               what produced the results in results/.

  [3] Full + exact-simulator ABC baseline - hours, and not needed.

Enter 1, 2 or 3 [1]:
```

Option 3 exists only for completeness: the reported results were produced
without the exact-simulator baseline (`results/logs/experiment_config.json`
records `"with_sim": false`), because this study compares the two surrogates to
each other rather than to the exact sampler.

**From an IDE (Spyder, VS Code, PyCharm, IDLE):** open `run_all.py` and press
**Run**. Answer the question in the console pane, and that is the whole job.
In Spyder the answer goes in the IPython console at the bottom right — click
into it, type `2`, press Enter.

**From a terminal:** `cd` into this folder and run `python run_all.py`
(`python3` on Mac/Linux). To skip the question — a server job, say — pass
`--quick`, `--full` or `--with-sim`.

The pipeline runs with whatever interpreter your IDE is using, so the packages
in `requirements.txt` have to be installed there. If any are missing the script
says which, and offers to install them for you; answer `y` once and it handles
it.

Each step prints its output to the console as it runs, and the run stops with
the error visible if a step fails.

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

`run_all.py` is only a wrapper around these five commands, so you can run them
yourself instead. About four minutes in total:

```bash
python network/train.py --data data/slow_data_3D.csv --seed 0
python tests/validate_simulator.py --quick
python abc/run_experiments.py --reps 16 --nmcmc 3000 --burnin 1000 --ns 4 --J-grid 100 --no-sim
python abc/mcse.py
python figures/make_figures.py
```

That reproduces the reported results. Drop `--no-sim` from the third command to
include the exact-simulator baseline instead (hours, and not needed). On Windows
use backslashes in the script paths (`python network\train.py ...`).

## 3. Check it worked

Two independent checks.

**(a) The fit.** Training is deterministic:

```bash
python network/train.py --seed 0
```

You should see:

```
conformal sd_scale = 1.0185
[test ] n= 4000  mse_mean=3.628e-04 (1.05x its 2-rep floor)  mse_obs=7.062e-04  95%cover=0.955
```

**(b) The simulator and the data** (~40 seconds):

```bash
python tests/validate_simulator.py --quick
```

Every line should print PASS. This confirms the two-stage simulator reduces
exactly to the constant-rate model in both limiting cases, and re-derives from
the data itself which mutation-time convention the ground truth was generated
under.

(The very last digits in `results/model/surrogate_metrics.json` may differ
between machines — ordinary floating-point variation, not a problem.)

## 4. What you'll see while it runs

`run_all.py` asks which run you want, then prints four labelled steps. Real
output, abbreviated:

```
Interpreter: C:\Users\you\anaconda3\python.exe
Folder:      C:\...\Models\3D

Which run do you want?

  [1] Quick  - a smoke test, roughly 1.5 minutes. ...
  [2] Full   - the reported settings: 16 replicates, a few minutes. ...
  [3] Full + exact-simulator ABC baseline - hours, and not needed. ...

Enter 1, 2 or 3 [1]: 2
=== FULL RUN: 16 replicates, reported settings ===

--- [1/4] Training the surrogate (~30 seconds) ---
train n=10000  val n=6000  test n=4000  features=['log10p1', 'log10p2', 'tau']
E[within-design variance] = 6.903e-04  ->  irreducible floor on the 2-replicate test target = 3.451e-04
conformal sd_scale = 1.0185
[test ] n= 4000  mse_mean=3.628e-04 (1.05x its 2-rep floor)  95%cover=0.955

--- [1b/4] Validating the simulator and the ground truth ---
  [PASS] tau >= tp -> stage 1 only (p1)  [parent, exact]
  [PASS] tau <= 0  -> stage 2 only (p2)  [parent, exact]
  [PASS] tau >> tp -> stage 1 only (p1)  [offspring, in distribution]
  [PASS] tau <= 0  -> stage 2 only (p2)  [offspring, in distribution]
  [PASS] convention gap is measurable and of the documented magnitude
  [PASS] ground truth was generated with mut_time = 'offspring'
all checks passed

--- [2/4] Estimator comparison: parameter recovery ---
48 tasks on 12 workers (without the exact ABC-MCMC baseline)
  [40/48] elapsed 3.0m  eta 0.6m
  [48/48] elapsed 3.0m  eta 0.0m

--- [3/4] Monte Carlo standard errors ---
--- [4/4] Figures ---
Done in 3.6 min. Everything written to results/:
```

Each step also echoes the exact command it is running, so if one fails you can
re-run that single command on its own to see the error again.

**All six validation checks in step 1b should print PASS.** If any prints FAIL,
stop and send me the output — that step exists to catch exactly the class of
problem that would otherwise produce quietly wrong tables.

Step 2 prints a running task count with an ETA. The whole thing is a few
minutes, so there are no long silent stretches here (unlike the 1-D study).

## 5. What comes out

Everything lands in `results/`:

| File | What it is |
|---|---|
| `results/tables/TABLES.md` | parameter recovery, formatted for reading |
| `results/tables/mcse.md` | Monte Carlo standard errors — which differences are real and which are noise |
| `results/figures/*.png` | all figures |
| `results/model/` | the trained surrogate and its fit metrics |

`results/` is **already populated** with a committed full run, so you can read
all of the above without running anything.

## 6. What the model does, and what it found

The mutation rate is no longer constant: it jumps from `p1` to `p2` at an
unknown time `τ`, and all three parameters are estimated **jointly** from one
scalar summary statistic.

**That statistic is the fourth root**, `S = meanᵢ (Xᵢ/Zᵢ)^(1/4)`, following
Sec. 2.2 of the paper — the square root is used for the constant-rate study, but
the paper switches to the fourth root for the two-stage model.

A neural network predicts the mean **and the variance** of `log10 S` from
`(log10 p1, log10 p2, τ)`, and replaces the simulator inside an ABC-MCMC
sampler. Three surrogates are compared:

| Method | What it is |
|---|---|
| `DNN-ABC` | this network |
| `GPS-ABC` | a Gaussian process, strengthened: anisotropic kernel, log-scaled inputs |
| `GPS-ABC-ref` | a Gaussian process matching `matlab/demoGPS_fluc_exp2.m` exactly |

Both GPs are reported because the strengthened one is better than the reference
on every axis, so quoting only it would overstate the published baseline.

**The headline result is negative and is reported as such.** DNN-ABC and
GPS-ABC are statistically tied in seven of nine parameter-by-truth comparisons;
the other two favour GPS-ABC and none favours the network. What binds here is
not surrogate error but the model's own identifiability: `p1` and `τ` are weakly
determined by a single scalar summary, so a better surrogate cannot help.

## 7. Regenerating the ground-truth data (not needed)

`data/slow_data_3D.csv` (2,000 design points × 10 replicates, 35 MB) is
included. It came from the exact cell-by-cell R simulator and takes hours on a
compute server; `run_on_server.sh` does it. The R source is in `../../RCode/`
and the reference MATLAB it was ported from is in `matlab/`.

## 8. Folder map

```
3D/
├── run_all.py       the entry point — press Run in an IDE, or `python run_all.py`
├── HOW_TO_RUN.md    this file
├── requirements.txt Python packages
├── data/            ground-truth data (included)
├── network/         the network, its training, and the architecture search
├── abc/             two-stage simulator, the three surrogates, ABC-MCMC sampler
├── matlab/          your original MATLAB, byte-identical: the two-stage
│                    simulator and the Study 2 GP driver
├── tests/           simulator checks + ground-truth provenance check
├── figures/         figure generation
└── results/         all outputs (already populated)
```

Every file opens with a comment block explaining what it does and which part of
the method it implements.

**On the imports at the top of the Python files:** names like `model`,
`surrogates`, `simulator` and `paths` are files in this folder, not installable
packages. Each script adds its sibling folders to `sys.path` itself, so
`python network/train.py` works from any directory as long as you have the whole
`3D/` folder.
