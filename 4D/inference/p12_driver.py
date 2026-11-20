"""Run the budget experiment (p12_budget.py) over a grid of (n, seed) with a wide worker pool that fills free cores.
Each worker calls dyncores.wait_for_core() before taking a task (on Linux it blocks until a core is idle; on macOS, which has no
/proc/stat, it returns at once, so --workers caps the load). Longest tasks (largest n) go first.
    python inference/p12_driver.py --data data/slow_data_4D.csv --out results/budget_k1000 --seeds 0 1 2 --K 1000 --workers 6
"""
import argparse, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dyncores import wait_for_core, wide_pool_size


def run(task):
    n, s, a = task
    out = Path(a["out"]) / f"p12_n{n}_s{s}.json"
    if out.exists(): return f"skip {out.name}"
    wait_for_core(reserve=a["threads"])
    with open(Path(a["out"]) / f"log_n{n}_s{s}.txt", "w") as lg:
        r = subprocess.run([sys.executable, str(HERE / "p12_budget.py"), "--data", a["data"], "--n", str(n), "--seed", str(s), "--K", str(a["K"]),
                            "--threads", str(a["threads"]), "--out", str(out)], stdout=lg, stderr=subprocess.STDOUT)
    return f"{out.name} exit {r.returncode}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[8000, 3000, 1500, 500]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=1000); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--workers", type=int, default=wide_pool_size())
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True); cfg = dict(out=a.out, data=a.data, K=a.K, threads=a.threads)
    tasks = [(n, s, cfg) for n in sorted(a.ns, reverse=True) for s in a.seeds]
    with Pool(min(a.workers, len(tasks))) as p:
        for msg in p.imap_unordered(run, tasks): print(msg, flush=True)


if __name__ == "__main__":
    main()
