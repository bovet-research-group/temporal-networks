# temporal-networks

<!-- header-start -->
A library for analyzing temporal networks.
<!-- header-end -->

## Installation

<!-- installation-start -->
The `tempnet` package requires Python 3.10 or higher.

Install from PyPI:

```bash
pip install tempnet
```

To install directly from source:

```bash
pip install git+https://github.com/bovet-research-group/temporal-networks.git
```
<!-- installation-end -->

## Development

Start by getting a local copy of the repository:

```bash
git clone https://github.com/bovet-research-group/temporal-networks.git
cd temporal-networks
```
Initialize a virtual environment and install the project in editable mode
together with all development dependencies (`testing` + `docs` + `benchmarks`
groups; requires pip ≥ 25.1):

```bash
pip install -e . --group dev
```

Alternatively, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Running the tests

```bash
pytest
```

Tests marked `network` download data from Zenodo and are skipped by default.
Run them explicitly when you have internet access:

```bash
pytest -m network
```

### Building the documentation locally

```bash
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

### Running the benchmarks

Performance benchmarks are managed with
[ASV (airspeed velocity)](https://asv.readthedocs.io/) and live in
`benchmarks/`.

Benchmarks are meant to be run locally (never in CI — shared runners produce
meaningless numbers). In an environment with the `benchmarks` group installed:

```bash
asv run                 # benchmark the latest commit on main
asv publish             # generate the HTML report in .asv/html
asv preview             # serve the report locally
```

Alternatively, with `uv`:

```bash
uv run asv run          # benchmark the latest commit on main
uv run asv publish      # generate the HTML report in .asv/html
uv run asv preview      # serve the report locally
```

Results accumulate in `.asv/results/`, so successive runs on different
commits build up a performance history. The raw results are **committed to
the repository** (only `.asv/env` and `.asv/html` are git-ignored): after a
benchmark session, commit the new files under `.asv/results/`. Any clone can
then regenerate the full browsable history with `asv publish && asv preview`.

Note that results are stored per machine (`.asv/results/<machine>/`), so
timings are only comparable within the same machine's series — ASV accounts
for this in the report.
