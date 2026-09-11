"""Run the whole 1-D pipeline: press Run in an IDE (Spyder, VS Code, PyCharm, IDLE).

There are no command-line arguments to type. The script asks, in the console,
whether you want the quick smoke test or the full paper-scale run, and then
does everything: trains the surrogate, runs the estimator comparison, computes
Monte Carlo standard errors, and regenerates every figure.

Everything runs with whatever interpreter the IDE is using (the one printed at
startup), so the packages in requirements.txt need to be installed in that
environment. If any are missing the script offers to install them for you.

If you do run this from a terminal and want to skip the question, pass --quick
or --full.
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
    """Ask quick-vs-full in the console. Falls back to quick if nothing can answer."""
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
            # No console to ask on (piped, or run by a scheduler). Quick is the
            # safe default: it cannot burn hours by accident.
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

    # Unbuffered, so the progress counter in step 2 appears live rather than in
    # one lump at the end. Line-by-line print() keeps it visible in Spyder's
    # console, which does not show a subprocess's raw output stream.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
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
        # Few workers on purpose: each one fits its own GP baseline at startup,
        # which with only 2 tasks would otherwise dominate the runtime.
        experiment_args = ["--reps", 2, "--nmcmc", 120, "--burnin", 40, "--ns", 6,
                           "--p-grid", "1e-2", "--J-grid", 10, "--workers", 2]
        # The posterior figure runs the exact simulator and ignores every
        # setting above, so it needs shrinking separately.
        figure_args = ["--quick"]
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
