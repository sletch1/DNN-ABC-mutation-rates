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

Then run the whole pipeline — about four minutes:

```powershell
python network\train.py --seed 0
python tests\validate_simulator.py --quick
python abc\run_experiments.py --reps 16 --nmcmc 3000 --burnin 1000 --no-sim
python abc\mcse.py
python figures\make_figures.py
```

That reproduces the reported results. Add `--with-sim` to the third command to
include the exact-simulator baseline instead (hours, and not needed).

### Windows (Git Bash or WSL) — if you prefer the one-command version

`run_all.sh` needs a bash shell. Install
[Git for Windows](https://git-scm.com/download/win), right-click in this folder →
"Git Bash Here", then:

```bash
./run_all.sh             # full run, reported settings (~4 minutes)
./run_all.sh --quick     # ~1.5 minutes, just to check it runs
```

If you get `bad interpreter` or `\r: command not found`, Git checked the script
out with Windows line endings. Fix it with:
`git config core.autocrlf input` and re-clone, or run `dos2unix run_all.sh`.

### Mac / Linux

```bash
./run_all.sh             # full run, reported settings (~4 minutes)
./run_all.sh --quick     # ~1.5 minutes, just to check it runs
                         # (~2 the very first time: see note below)
./run_all.sh --with-sim  # also run the exact-simulator baseline (hours)
```

If you get a permissions error: `chmod +x run_all.sh`, then retry.

**First run only:** the script creates its own `.venv/` and downloads the
packages, which adds roughly two minutes. Every run after that reuses it. Both
cold-start paths were tested end to end and complete cleanly.

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

## 4. What comes out

Everything lands in `results/`:

| File | What it is |
|---|---|
| `results/tables/TABLES.md` | parameter recovery, formatted for reading |
| `results/tables/mcse.md` | Monte Carlo standard errors — which differences are real and which are noise |
| `results/figures/*.png` | all figures |
| `results/model/` | the trained surrogate and its fit metrics |

`results/` is **already populated** with a committed full run, so you can read
all of the above without running anything.

## 5. What the model does, and what it found

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

## 6. Regenerating the ground-truth data (not needed)

`data/slow_data_3D.csv` (2,000 design points × 10 replicates, 35 MB) is
included. It came from the exact cell-by-cell R simulator and takes hours on a
compute server; `run_on_server.sh` does it. The R source is in `../../RCode/`
and the reference MATLAB it was ported from is in `matlab/`.

## 7. Folder map

```
3D/
├── run_all.sh       the entry point
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
