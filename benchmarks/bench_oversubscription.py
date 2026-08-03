"""Oversubscription diagnostics for the ``tempnet.faster_expm`` module.

``track_*`` benchmarks report raw numbers (not timings). They answer three
questions that explain *why* a multiprocessing wrapper can be slower than
the plain scipy call:

1. How many BLAS threads does scipy's ``expm`` already use under the hood?
   (``track_blas_threads``) — if this is > 1, wrapping in multiprocessing
   competes with the existing low-level parallelism.
2. How many OS processes does a pooled implementation actually spawn?
   (``track_peak_process_count``) — together with (1), the *potential*
   concurrency is roughly ``processes x blas_threads``; if that exceeds
   the CPU count, the run can be oversubscribed.
3. Does the run *actually* over-commit the cores?
   (``track_oversubscription_factor``) — peak number of runnable threads
   across the whole process tree divided by the CPU count. 1.0 means the
   cores are exactly saturated, > 1.0 means the OS is time-slicing
   runnable threads (real oversubscription), well below 1.0 means the
   parallelism is underutilized.
"""

import glob
import os
import threading
import time

import psutil
from scipy.sparse.linalg import expm
from threadpoolctl import threadpool_info

try:
    from tempnet.faster_expm import compute_subspace_expm_parallel
except ImportError:
    # pre-refactor layout (main); keeps `asv continuous` working.
    from tempnet.parallel_expm import compute_subspace_expm_parallel

from .common import block_laplacian

NCPU = os.cpu_count() or 1


def _runnable_threads_proc(pids):
    """Count runnable ('R') threads via ``/proc/<pid>/task/*/stat`` (Linux).

    'R' covers both *running* and *runnable-but-waiting-for-a-core*
    threads, which is exactly the over-commit signal: more runnable
    threads than cores means the scheduler is time-slicing.
    """
    n = 0
    for pid in pids:
        for stat in glob.glob(f"/proc/{pid}/task/*/stat"):
            try:
                with open(stat) as fh:
                    # The state char is the first field after the comm
                    # field; comm may contain spaces/parens, so split on
                    # the *last* ')'.
                    state = fh.read().rpartition(")")[2].split()[0]
            except OSError:
                continue  # thread exited between glob and read
            if state == "R":
                n += 1
    return n


def _peak_while(sample_fn, work_fn, interval=0.01):
    """Run ``work_fn()`` while sampling ``sample_fn()`` from a watcher
    thread; return the peak sampled value.
    """
    peak = 0
    done = threading.Event()

    def watch():
        nonlocal peak
        while not done.is_set():
            try:
                peak = max(peak, sample_fn())
            except psutil.Error:
                pass
            time.sleep(interval)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        work_fn()
    finally:
        done.set()
        watcher.join()

    return peak


class TrackBlasThreads:
    """Report the BLAS thread-pool size scipy's expm has at its disposal.

    If this is > 1, plain scipy ``expm`` is already parallelized under
    the hood, and every Pool worker in the pooled implementations brings
    its own pool of this size — the source of potential oversubscription
    (``track_cpu_count`` is reported alongside for the comparison).
    """

    unit = "threads"

    def setup(self):
        # trigger BLAS initialization so threadpool_info sees the pools
        L = block_laplacian(200, 1)
        expm(-L)

    def track_blas_threads(self):
        infos = threadpool_info()
        if not infos:
            return 1
        return max(info["num_threads"] for info in infos)

    def track_cpu_count(self):
        return NCPU


class TrackPeakProcessCount:
    """Peak number of live child processes during a pooled expm run.

    Together with ``track_blas_threads`` this gives the *potential*
    concurrency (``processes x blas_threads``); compare against the CPU
    count to see whether a run *can* be oversubscribed. Whether it
    actually is, is measured by ``TrackOversubscriptionFactor``.
    """

    unit = "processes"
    params = sorted({1, 2, 4, NCPU})
    param_names = ["nproc"]
    timeout = 600

    def setup(self, nproc):
        self.A = -block_laplacian(2000, 10)

    def track_peak_process_count(self, nproc):
        proc = psutil.Process()

        return _peak_while(
            lambda: len(proc.children(recursive=True)),
            lambda: compute_subspace_expm_parallel(
                self.A.copy(),
                nproc=nproc,
                normalize_rows=False,
                verbose=False,
            ),
        )


class TrackOversubscriptionFactor:
    """Peak runnable threads across the process tree / cpu_count.

    1.0 = cores exactly saturated; > 1.0 = over-committed (the OS is
    time-slicing runnable threads); values well below 1.0 = the
    parallelism is underutilized.

    Linux-only (reads ``/proc``); skipped elsewhere.
    """

    unit = "x cpu_count"
    params = sorted({1, 2, 4, NCPU})
    param_names = ["nproc"]
    timeout = 600

    def setup(self, nproc):
        if not glob.glob("/proc/self/task/*/stat"):
            raise NotImplementedError("requires Linux /proc")
        self.A = -block_laplacian(2000, 10)

    def track_oversubscription_factor(self, nproc):
        proc = psutil.Process()

        def sample():
            pids = [proc.pid]
            pids += [c.pid for c in proc.children(recursive=True)]
            return _runnable_threads_proc(pids)

        peak = _peak_while(
            sample,
            lambda: compute_subspace_expm_parallel(
                self.A.copy(),
                nproc=nproc,
                normalize_rows=False,
                verbose=False,
            ),
        )
        return round(peak / NCPU, 2)
