"""Run the whole 3-D two-stage pipeline: press Run in an IDE, or pass
--quick/--full/--with-sim from a terminal. Asks which run, installs missing
packages if needed, then trains the surrogate, validates the simulator, runs
the estimator comparison, computes MCSEs, and regenerates every figure.
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

  [2] Full   - the reported settings without the exact-simulator baseline:
               16 replicates, a few minutes. Reproduces every method in
               results/ except the ABC-MCMC (exact) row/column.

  [3] Full + exact-simulator ABC baseline - several hours (measured: ~4.2h on
               a 14-core laptop, 10 workers). Reproduces the ABC-MCMC (exact)
               row in TABLES.md/paper Table 9; costly because it calls the
               true simulator every MCMC iteration. To backfill it onto an
               existing run instead of rerunning everything, use
               abc/add_exact.py (run_all.py doesn't do this, unlike
               abc/add_npe.py for NPE).
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
    # Cheap sanity checks (simulator reduces correctly, mutation-time convention
    # matches the ground truth) -- fails loudly rather than producing wrong tables.
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
