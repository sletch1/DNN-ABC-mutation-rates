"""Dynamic core use for shared multi-core servers.

A job starts a wide worker pool; before taking each task a worker calls `wait_for_core()`, which blocks until the machine has
an idle core (measured from /proc/stat over a short window, so other users' and other jobs' load counts). The job therefore
fills whatever the other work leaves free and backs off when it is needed, with no fixed worker count to tune.
No-op on machines without /proc/stat (macOS): returns immediately.
"""
import os, random, time

def _cpu():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + (v[4] if len(v) > 4 else 0)   # total, idle+iowait

def idle_cores(window=0.7):
    try:
        t0, i0 = _cpu(); time.sleep(window); t1, i1 = _cpu()
    except OSError:
        return 1e9
    return (os.cpu_count() or 1) * (i1 - i0) / max(1, t1 - t0)

def wait_for_core(reserve=1.0, poll=15, jitter=20, max_wait=3600):
    """Block until at least `reserve` cores are idle (random jitter first so simultaneous workers do not all start at once)."""
    time.sleep(random.random() * jitter)
    t = time.time()
    while idle_cores() < reserve and time.time() - t < max_wait:
        time.sleep(poll * (0.5 + random.random()))

def wide_pool_size(cap=30):
    return min(cap, os.cpu_count() or 1)
