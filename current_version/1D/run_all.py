"""Run the whole 1-D pipeline: press Run in an IDE, or pass --quick/--full
from a terminal. Asks quick-smoke-test vs. full-paper-scale, installs missing
packages if needed, then trains the surrogate, runs the estimator comparison,
computes MCSEs, and regenerates every figure.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable  # the IDE's own interpreter, not .venv

REQUIRED = {  # import name -> pip name
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "torch": "torch",
}

MENU = """
Which run do you want?

  [1] Quick  - a smoke test, roughly 3 minutes. Confirms the pipeline works
               end to end. Its numbers are noisy: do not read results off it.

  [2] Full   - the paper's settings: 40 replicates, 3 mutation rates, 3 culture
               counts. Several hours (see HOW_TO_RUN.md section 6). You very
               likely do not need this -- results/ already holds a full run.
"""


def ask_quick():
    """Ask quick-vs-full in the console. Defaults to quick if unanswerable
    (piped input, no console) so it can't silently burn hours."""
    argv = [a.lower() for a in sys.argv[1:]]
    if "--quick" in argv:
        return True
    if "--full" in argv:
        return False

    print(MENU, flush=True)
    while True:
        try:
            answer = input("Enter 1 or 2 [1]: ").strip()
        except (EOFError, OSError):
            print("(no input available -- defaulting to the quick run)", flush=True)
            return True
        if answer in ("", "1"):
            return True
        if answer == "2":
            return False
        print("Please type 1 or 2.", flush=True)


def check_packages():
    """Make sure the IDE's interpreter has what the pipeline needs."""
    import importlib.util
    missing = [pip for mod, pip in REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return

    print("\nMissing packages: " + ", ".join(missing), flush=True)
    try:
        answer = input("Install them now into the interpreter above? [y/N]: ").strip().lower()
    except (EOFError, OSError):
        answer = "n"
    if answer in ("y", "yes"):
        subprocess.check_call([PY, "-m", "pip", "install", "-r",
                               str(HERE / "requirements.txt")])
        return
    raise SystemExit(
        "\nInstall them yourself with:\n\n"
        f'    "{PY}" -m pip install -r "{HERE / "requirements.txt"}"\n\n'
        "then run this file again."
    )


def run(label, script, args=()):
    """Run one pipeline step, echoing its output into the IDE console as it goes."""
    print("\n--- " + label + " ---", flush=True)
    cmd = [PY, str(HERE / script), *[str(a) for a in args]]
    print("  " + " ".join(cmd), flush=True)

    # Unbuffered so progress prints live instead of in one lump at the end.
    # Thread count capped: these networks are tiny, so PyTorch's default of
    # one thread/core causes thread-spawn overhead to dominate on a many-core
    # machine (measured: a 30s training step became 10+ min at 800%+ CPU).
    n_threads = str(min(4, os.cpu_count() or 1))
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=n_threads, MKL_NUM_THREADS=n_threads)
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"\n{script} failed (exit code {proc.returncode}). "
                         f"The error is in the output just above.")


def main():
    print("Interpreter: " + PY, flush=True)
    print("Folder:      " + str(HERE), flush=True)

    quick = ask_quick()
    check_packages()

    if quick:
        print("\n=== QUICK MODE: pipeline check only, NOT paper-scale results ===",
              flush=True)
        # workers=2: each worker fits its own GP at startup, not worth it for 2 tasks.
        experiment_args = ["--reps", 2, "--nmcmc", 120, "--burnin", 40, "--ns", 6,
                           "--p-grid", "1e-2", "--J-grid", 10, "--workers", 2]
        figure_args = ["--quick"]  # posterior figure ignores the args above; shrink separately
    else:
        print("\n=== FULL RUN: 40 replicates, paper settings (expect several hours) ===",
              flush=True)
        experiment_args = ["--reps", 40, "--nmcmc", 600, "--burnin", 250, "--ns", 6,
                           "--p-grid", "1e-4", "1e-3", "1e-2",
                           "--J-grid", 10, 50, 100]   # default workers: all cores but two
        figure_args = []

    t0 = time.time()
    run("[1/4] Training the surrogate (fast, well under a minute)",
        "network/train.py")
    run("[2/4] Estimator comparison: Tables 1, 2 and 3 (this is the long one)",
        "abc/run_experiments.py", experiment_args)
    run("[3/4] Monte Carlo standard errors", "abc/mcse.py")
    run("[4/4] Figures", "figures/make_figures.py", figure_args)
    run("[4/4] Architecture diagram", "network/gen_architecture_svg.py")

    print("\n===========================================================")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Everything written to results/:")
    print("  results/tables/TABLES.md   Tables 1-3, formatted for reading")
    print("  results/tables/mcse.md     which differences are real vs. noise")
    print("  results/figures/*.png      all result figures")
    print("  results/model/             the trained surrogate + its fit metrics")
    print("===========================================================", flush=True)


if __name__ == "__main__":
    main()
