"""Timing and peak-memory benchmarks for the ``tempnet.faster_expm`` module.

Benchmarked implementations, all computing (a variant of) ``expm(-L)``:

- ``scipy.sparse.linalg.expm`` — baseline
- ``compute_subspace_expm`` — serial, per-connected-component
- ``compute_subspace_expm_parallel`` — per-component with a worker pool
- ``compute_parallel_expm`` — column-wise ``expm_multiply`` with a pool
- ``sparse_lapl_expm`` — public entry point (dense and sparse branches)

Design notes
------------
- Classes are split by parameter shape: serial implementations have no
  ``nproc`` axis; pooled implementations sweep it. This avoids fake
  "skipped" parameter combinations in the ASV report.
- The ``nproc`` parameter axis is the core of the oversubscription analysis:
  scipy's ``expm`` is already low-level parallelized (BLAS threads), so
  wrapping it in multiprocessing can be a *slowdown*. If ``nproc > 1`` is
  slower than ``nproc = 1`` for a given size, that is oversubscription.
- The ``*SingleThreadedBlas`` classes re-run the same sweeps with BLAS
  pinned to a single thread (via threadpoolctl), separating "multiprocessing
  loses against multithreaded BLAS" from "pure multiprocessing overhead".
- ``peakmem_*`` benchmarks are run by ASV in independent processes from the
  timing runs and pin BLAS to one thread. Parallel peak-memory benchmarks
  still sweep ``nproc``; the pinning only prevents nested BLAS thread pools
  from turning memory measurements into oversubscription measurements.
  Note: peak memory covers the *main* process only; memory of Pool workers
  is not captured (use ``memray --follow-forks`` if needed).
- Benchmarks are meant to be run locally (``asv run``), never in CI.
"""

import os

from scipy.sparse.linalg import expm
from threadpoolctl import threadpool_limits

try:
    from tempnet.faster_expm import (
        compute_parallel_expm,
        compute_subspace_expm,
        compute_subspace_expm_parallel,
        sparse_lapl_expm,
    )
except ImportError:
    # pre-refactor layout (main): functions split across two modules;
    # keeps `asv continuous main HEAD` working across the rename.
    from tempnet.parallel_expm import (
        compute_parallel_expm,
        compute_subspace_expm_parallel,
    )
    from tempnet.temporal_network import (
        compute_subspace_expm,
        sparse_lapl_expm,
    )

from .common import block_laplacian, pretty_name

NCPU = os.cpu_count() or 1

SIZES = [500, 2000, 8000]
N_COMPONENTS = [1, 10, 100]
NPROCS = sorted({1, 2, 4, NCPU})


class _SerialBase:
    """Setup for serial implementations (no nproc axis)."""

    params = (SIZES, N_COMPONENTS)
    param_names = ["size", "n_components"]

    # generous per-run limit; large dense expm on 8000 nodes is slow
    timeout = 600

    def setup(self, size, n_components):
        self.L = block_laplacian(size, n_components)
        self.A = -self.L


class _ParallelBase:
    """Setup for pooled implementations (sweeps nproc)."""

    params = (SIZES, N_COMPONENTS, NPROCS)
    param_names = ["size", "n_components", "nproc"]

    timeout = 600

    def setup(self, size, n_components, nproc):
        self.L = block_laplacian(size, n_components)
        self.A = -self.L


class TimeExpmSerial(_SerialBase):
    """Wall-clock timings of the serial implementations (default BLAS).

    Shows whether the per-connected-component decomposition
    (``compute_subspace_expm``) beats plain scipy ``expm``: decomposition
    should win for many small components and degenerate to plain ``expm``
    for a single component.
    """

    @pretty_name("SciPy sparse expm baseline")
    def time_scipy_expm(self, size, n_components):
        expm(self.A)

    @pretty_name("Serial subspace expm by connected component")
    def time_compute_subspace_expm(self, size, n_components):
        compute_subspace_expm(self.A.copy(), normalize_rows=False)


class TimeExpmParallel(_ParallelBase):
    """Wall-clock timings of the pooled implementations (default BLAS).

    The ``nproc`` axis is the oversubscription probe: scipy's ``expm``
    is already parallelized at the BLAS level, so each pool worker
    brings its own BLAS thread pool. If ``nproc > 1`` is *slower* than
    ``nproc = 1`` for a given size, the workers are fighting over cores
    instead of adding throughput (see ``bench_oversubscription`` for the
    diagnostics that attribute this).
    """

    @pretty_name("Parallel subspace expm")
    def time_compute_subspace_expm_parallel(self, size, n_components, nproc):
        compute_subspace_expm_parallel(
            self.A.copy(),
            nproc=nproc,
            normalize_rows=False,
        )

    @pretty_name("Parallel column-wise expm_multiply")
    def time_compute_parallel_expm(self, size, n_components, nproc):
        compute_parallel_expm(
            self.A.copy(),
            nproc=nproc,
            normalize_rows=False,
        )

    @pretty_name("sparse_lapl_expm sparse branch")
    def time_sparse_lapl_expm_sparse(self, size, n_components, nproc):
        sparse_lapl_expm(
            self.L.copy(),
            fact=1.0,
            dense_expm=False,
            nproc=nproc,
        )


class TimeExpmSerialSingleThreadedBlas(_SerialBase):
    """Serial implementations with BLAS pinned to one thread.

    Baseline for the BLAS-pinned comparison: together with
    ``TimeExpmSerial`` it quantifies how much of plain ``expm``'s speed
    comes from BLAS-level multithreading on this machine. The dense
    ``sparse_lapl_expm`` path is only timed here because the default-BLAS
    version is dominated by scipy/BLAS scheduling noise and can create
    false positives in branch comparisons even when the tempnet code is
    identical.
    """

    @pretty_name("SciPy sparse expm baseline (BLAS=1)")
    def time_scipy_expm(self, size, n_components):
        with threadpool_limits(1):
            expm(self.A)

    @pretty_name("Serial subspace expm (BLAS=1)")
    def time_compute_subspace_expm(self, size, n_components):
        with threadpool_limits(1):
            compute_subspace_expm(self.A.copy(), normalize_rows=False)

    @pretty_name("sparse_lapl_expm dense branch (BLAS=1)")
    def time_sparse_lapl_expm_dense(self, size, n_components):
        with threadpool_limits(1):
            sparse_lapl_expm(self.L.copy(), fact=1.0, dense_expm=True)


class TimeExpmParallelSingleThreadedBlas(_ParallelBase):
    """Pooled implementations with BLAS pinned to one thread.

    Comparing against ``TimeExpmParallel`` separates two effects:

    - if multiprocessing only wins here, it merely competes with BLAS
      threads (oversubscription in the default setup);
    - if multiprocessing loses even here, the pool overhead itself
      outweighs the parallelism gain.
    """

    @pretty_name("Parallel subspace expm (BLAS=1)")
    def time_compute_subspace_expm_parallel(self, size, n_components, nproc):
        with threadpool_limits(1):
            compute_subspace_expm_parallel(
                self.A.copy(),
                nproc=nproc,
                normalize_rows=False,
            )

    @pretty_name("Parallel column-wise expm_multiply (BLAS=1)")
    def time_compute_parallel_expm(self, size, n_components, nproc):
        with threadpool_limits(1):
            compute_parallel_expm(
                self.A.copy(),
                nproc=nproc,
                normalize_rows=False,
            )


class PeakMemExpmSerial(_SerialBase):
    """Peak memory of the serial implementations (main process only).

    Dense ``expm`` materializes an ``n x n`` array; the subspace
    decomposition only ever holds one component's dense block at a time,
    so its peak should drop sharply as ``n_components`` grows. BLAS is
    pinned to one thread because these benchmarks measure allocation
    behaviour, not CPU throughput.
    """

    @pretty_name("Peak memory: SciPy sparse expm")
    def peakmem_scipy_expm(self, size, n_components):
        with threadpool_limits(1):
            expm(self.A)

    @pretty_name("Peak memory: serial subspace expm")
    def peakmem_compute_subspace_expm(self, size, n_components):
        with threadpool_limits(1):
            compute_subspace_expm(self.A.copy(), normalize_rows=False)

    @pretty_name("Peak memory: sparse_lapl_expm dense branch")
    def peakmem_sparse_lapl_expm_dense(self, size, n_components):
        with threadpool_limits(1):
            sparse_lapl_expm(self.L.copy(), fact=1.0, dense_expm=True)


class PeakMemExpmParallel(_ParallelBase):
    """Peak memory of the pooled implementations.

    Measures the *main* process only: the shared ``RawArray`` copies of
    the input plus result assembly. Memory of the Pool workers is not
    captured by ASV's peakmem (use ``memray --follow-forks`` for a full
    per-worker picture). The ``nproc`` axis is preserved, but BLAS is
    pinned to one thread so the measurement reflects process-level
    parallelism rather than nested BLAS oversubscription.
    """

    @pretty_name("Peak memory: parallel subspace expm")
    def peakmem_compute_subspace_expm_parallel(self, size, n_components,
                                               nproc):
        with threadpool_limits(1):
            compute_subspace_expm_parallel(
                self.A.copy(),
                nproc=nproc,
                normalize_rows=False,
            )

    @pretty_name("Peak memory: parallel column-wise expm_multiply")
    def peakmem_compute_parallel_expm(self, size, n_components, nproc):
        with threadpool_limits(1):
            compute_parallel_expm(
                self.A.copy(),
                nproc=nproc,
                normalize_rows=False,
            )

    @pretty_name("Peak memory: sparse_lapl_expm sparse branch")
    def peakmem_sparse_lapl_expm_sparse(self, size, n_components, nproc):
        with threadpool_limits(1):
            sparse_lapl_expm(
                self.L.copy(),
                fact=1.0,
                dense_expm=False,
                nproc=nproc,
            )


class TimeSparseCore:
    """``sparse_lapl_expm`` advantage on mostly-zero laplacians.

    Isolates the ``remove_nnz_rowcol`` optimization: only a small fraction
    of nodes participate in events, the rest are zero rows/cols.
    """

    params = (SIZES, [0.01, 0.1, 0.5])
    param_names = ["size", "active_fraction"]
    timeout = 600

    def setup(self, size, active_fraction):
        from .common import sparse_core_laplacian
        self.L = sparse_core_laplacian(size, active_fraction)

    @pretty_name("SciPy expm on mostly-zero laplacian")
    def time_scipy_expm(self, size, active_fraction):
        expm(-self.L)

    @pretty_name("sparse_lapl_expm trims inactive nodes")
    def time_sparse_lapl_expm_dense(self, size, active_fraction):
        sparse_lapl_expm(self.L.copy(), fact=1.0, dense_expm=True)


class TimeSyntheticNetwork:
    """Realistic workload: laplacian from a SynthTempNetwork simulation.

    The controlled sweeps above use idealized block Laplacians; this
    class benchmarks all variants on an actual temporal-network
    laplacian (the busiest time step of a seeded agent-based
    simulation). Temporal-network laplacians at a single time step are
    typically very sparse with many isolated nodes — the regime
    ``sparse_lapl_expm`` was written for.
    """

    timeout = 600

    def setup_cache(self):
        from .common import synthetic_temporal_laplacian
        return synthetic_temporal_laplacian(n_groups=3, n_per_group=20)

    @pretty_name("SciPy expm on synthetic-network laplacian")
    def time_scipy_expm(self, L):
        expm(-L)

    @pretty_name("Subspace expm on synthetic-network laplacian")
    def time_compute_subspace_expm(self, L):
        compute_subspace_expm(-L, normalize_rows=False)

    @pretty_name("sparse_lapl_expm dense on synthetic-network laplacian")
    def time_sparse_lapl_expm_dense(self, L):
        sparse_lapl_expm(L.copy(), fact=1.0, dense_expm=True)

    @pretty_name("sparse_lapl_expm sparse on synthetic-network laplacian")
    def time_sparse_lapl_expm_sparse(self, L):
        sparse_lapl_expm(L.copy(), fact=1.0, dense_expm=False)
