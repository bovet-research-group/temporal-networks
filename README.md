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
together with all development dependencies (`testing` + `docs` groups):

```bash
uv sync
```

Alternatively, with plain pip (requires pip ≥ 25.1):

```bash
pip install -e . --group dev
```

### Running the tests

```bash
uv run pytest
```

Tests marked `network` download data from Zenodo and are skipped by default.
Run them explicitly when you have internet access:

```bash
uv run pytest -m network
```

### Building the documentation locally

```bash
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

> **Note:** the `plot_02` gallery example downloads the mouse contact dataset
> from Zenodo during the build — this requires an internet connection and
> `zenodo-get` (included in the `docs` dependency group).
