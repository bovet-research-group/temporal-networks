# Benchmarks

This directory contains the ASV (airspeed velocity) benchmark suite for
`tempnet`. Benchmarks are intended for local or dedicated-machine runs, not
for shared CI runners where load and CPU throttling make timings noisy.

## Running benchmarks

Install the development dependencies first:

```bash
pip install -e . --group dev
```

Run the full benchmark suite:

```bash
asv run
```

Run a quick smoke check against the current environment:

```bash
asv run --python=same --quick
```

Publish and preview the HTML report:

```bash
asv publish
asv preview
```

Raw benchmark results are stored in `.asv/results/` and committed to the
repository. Generated environments and HTML output (`.asv/env/`, `.asv/html/`)
are not committed.

## Machine configuration

Benchmark sizes are configured in `machine_configs.json`. The file contains a
required `default` entry and optional machine-specific entries keyed by ASV
machine name:

```json
{
  "default": {
    "sizes": [500, 2000],
    "n_components": [1, 4, 8],
    "n_individuals": [12, 60],
    "t_ends": [100.0]
  },
  "jil_iMac_fedora": {
    "sizes": [100, 200, 400, 800, 1600, 3200],
    "n_components": [1, 4, 8, 32, 96],
    "n_individuals": [12, 60, 300],
    "t_ends": [100.0, 500.0]
  }
}
```

`Machine` in `common.py` reads the ASV machine name from
`~/.asv-machine.json`, merges the matching machine entry over `default`, and
creates `get_<key>()` accessors for every config key. Missing machine-specific
keys fall back to `default`.

To add a machine:

1. Register or inspect the ASV machine name with `asv machine`.
2. Add a matching entry to `benchmarks/machine_configs.json`.
3. Run `asv run --python=same --quick` to verify the configuration.

The number of worker processes is derived from CPU count, not from the JSON
config. It uses powers of two plus one less than the CPU count, for example:

- 4 CPUs -> `[1, 2, 3]`
- 8 CPUs -> `[1, 2, 4, 7]`
- 12 CPUs -> `[1, 2, 4, 8, 11]`

## Benchmark structure

### `bench_synth_network.py`

Measures the agent-based `SynthTempNetwork` simulation, construction of a
`ContTempNetwork` from simulated events, and laplacian computation on the
resulting temporal network. The parameters are `n_individuals` and `t_end`.

### `bench_expm.py`

Benchmarks matrix-exponential implementations used by temporal-network
dynamics on controlled block-diagonal Laplacians and one synthetic-network
laplacian. It compares scipy's sparse `expm`, serial subspace decomposition,
multiprocessing variants, and `sparse_lapl_expm` dense/sparse branches.

The BLAS-pinned classes (`*SingleThreadedBlas`) separate Python multiprocessing
overhead from oversubscription caused by combining multiprocessing with
BLAS-level threading. Peak-memory benchmarks are separate ASV runs and pin BLAS
to one thread.

### `bench_oversubscription.py`

Tracks diagnostics explaining parallel slowdown: BLAS thread count, logical
CPU count, peak worker process count, and peak runnable threads divided by CPU
count. Values above 1.0 for the oversubscription factor mean runnable threads
exceed available CPUs.

## Comparing branches

Use `asv continuous` for feature-branch comparisons:

```bash
asv continuous main HEAD
```

To compare against the exact branch point:

```bash
asv continuous $(git merge-base main HEAD) HEAD
```

Branch-comparison results are useful for pull-request review. The published
benchmark history should generally track the main branch.
