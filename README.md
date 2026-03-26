# temporal-networks
A library for analyzing temporal networks.

## Installation

The `tempnet` package requires Python versions of 3.10.0 or higher.
To ensure reproducible environments and precise dependency resolution, we
recommend using the open-source package manager `uv`:

Installation can be executed directly from the remote repository.
A local clone is not required.
After a virtual environment is initialized, the package is installed via the
following execution:

```bash
uv pip install git+https://github.com/bovet-research-group/temporal-networks.git
```

Dependencies required specifically for testing procedures are defined in the
project configuration.
These optional dependencies can be appended during the remote installation:

```bash
uv pip install "tempnet[testing] @ git+https://github.com/bovet-research-group/temporal-networks.git"
```

## Development

Start by getting a local copy of the repository:

```bash
git clone https://github.com/bovet-research-group/temporal-networks.git
cd temporal-networks
```

Initialize a virtual environment and install the project directly in editable
mode with:

```bash
uv sync
```
