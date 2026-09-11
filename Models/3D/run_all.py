"""Run the whole 3-D two-stage pipeline: press Run in an IDE (Spyder, VS Code, PyCharm, IDLE).

There are no command-line arguments to type. The script asks, in the console,
which run you want, and then does everything: trains the surrogate, validates
the simulator, runs the estimator comparison, computes Monte Carlo standard
errors, and regenerates every figure.

Everything runs with whatever interpreter the IDE is using (the one printed at
startup), so the packages in requirements.txt need to be installed in that
environment. If any are missing the script offers to install them for you.

If you do run this from a terminal and want to skip the question, pass --quick,
--full or --with-sim.
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

  [1] Quick  - a smoke test, roughly 1.5 minutes. Confirms the pipeline works
               end to end. Its numbers are noisy: do not read results off it.

  [2] Full   - the reported settings: 16 replicates, a few minutes. This is
               what produced the results in results/.

  [3] Full + exact-simulator ABC baseline - hours, and not needed. The reported
               results were produced without it (results/logs/experiment_config.json
               records "with_sim": false), because this study compares the two
               surrogates to each other rather than to the exact sampler.
"""


def ask_mode():
    """Ask which run to do. Returns (quick, with_sim). Defaults to quick if unanswerable."""
    argv = [a.lower() for a in sys.argv[1:]]
    if "--with-sim" in argv:
        return False, True
    if "--quick" in argv:
        return True, False
    if "--full" in argv:
        return False, False

    print(MENU, flush=True)
    while True:
        try:
            answer = input("Enter 1, 2 or 3 [1]: ").strip()
        except (EOFError, OSError):
            # No console to ask on (piped, or run by a scheduler). Quick is the
            # safe default: it cannot burn hours by accident.
            print("(no input available -- defaulting to the quick run)", flush=True)
            return True, False
        if answer in ("", "1"):
            return True, False
        if answer == "2":
            return False, False
        if answer == "3":
            return False, True
        print("Please type 1, 2 or 3.", flush=True)


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

    # Unbuffered, so progress appears live rather than in one lump at the end.
    # Line-by-line print() keeps it visible in Spyder's console, which does not
    # show a subprocess's raw output stream.
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

    quick, with_sim = ask_mode()
    check_packages()

    if quick:
        print("\n=== QUICK MODE: pipeline check only, NOT the reported results ===",
              flush=True)
        experiment_args = ["--reps", 2, "--nmcmc", 300, "--burnin", 100,
                           "--ns", 4, "--J-grid", 100, "--workers", 2]
    else:
        print("\n=== FULL RUN: 16 replicates, reported settings ===", flush=True)
        experiment_args = ["--reps", 16, "--nmcmc", 3000, "--burnin", 1000,
                           "--ns", 4, "--J-grid", 100]  # default workers: all cores but two

    if with_sim:
        print("=== exact-simulator baseline ENABLED (expect hours) ===", flush=True)
    else:
        experiment_args.append("--no-sim")

    t0 = time.time()
    run("[1/4] Training the surrogate (~30 seconds)",
        "network/train.py", ["--data", "data/slow_data_3D.csv", "--seed", 0])
    # Sanity checks before the comparison: that the two-stage simulator reduces
    # to the constant-rate one in both limits, and that the ground truth's
    # mutation-time convention is the one the pipeline assumes. Cheap, and it
    # fails loudly rather than producing quietly wrong tables.
    run("[1b/4] Validating the simulator and the ground truth",
        "tests/validate_simulator.py", ["--quick"])
    run("[2/4] Estimator comparison: parameter recovery",
        "abc/run_experiments.py", experiment_args)
    run("[3/4] Monte Carlo standard errors", "abc/mcse.py")
    run("[4/4] Figures", "figures/make_figures.py")
    run("[4/4] Architecture diagram", "network/gen_architecture_svg.py")

    print("\n===========================================================")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Everything written to results/:")
    print("  results/tables/TABLES.md   parameter recovery, formatted for reading")
    print("  results/tables/mcse.md     which differences are real vs. noise")
    print("  results/figures/*.png      all result figures")
    print("  results/model/             the trained surrogate + its fit metrics")
    print("===========================================================", flush=True)


if __name__ == "__main__":
    main()
