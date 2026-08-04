"""Shared, seeded matrix generators for the tempnet benchmarks.

All generators are deterministic (fixed seed or fully structural) so that
benchmark results are comparable across runs and commits.
"""

import numpy as np
from scipy.sparse import csc_matrix, diags


def pretty_name(name):
    """Set the human-readable ASV display name of a benchmark."""
    def decorate(func):
        func.pretty_name = name
        return func
    return decorate


def path_graph_laplacian(size):
    """Heat Laplacian of a path graph with `size` nodes (one component)."""
    nodes = np.arange(size - 1)
    rows = np.concatenate([nodes, nodes + 1])
    cols = np.concatenate([nodes + 1, nodes])
    data = np.ones(rows.shape[0], dtype=np.float64)

    adjacency = csc_matrix((data, (rows, cols)), shape=(size, size))
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return diags(degrees, format="csc") - adjacency


def block_laplacian(size, n_components):
    """Block-diagonal Laplacian: `n_components` path graphs of equal size.

    Isolates the connected-component decomposition advantage of the
    subspace expm implementations.
    """
    comp_size = size // n_components
    rows = []
    cols = []
    for c in range(n_components):
        offset = c * comp_size
        nodes = np.arange(comp_size - 1) + offset
        rows.append(nodes)
        cols.append(nodes + 1)
        rows.append(nodes + 1)
        cols.append(nodes)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.shape[0], dtype=np.float64)
    n = comp_size * n_components

    adjacency = csc_matrix((data, (rows, cols)), shape=(n, n))
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return diags(degrees, format="csc") - adjacency


def sparse_core_laplacian(size, active_fraction=0.1):
    """Laplacian with mostly zero rows/cols and a small active path-graph core.

    Isolates the `remove_nnz_rowcol` advantage of `sparse_lapl_expm`: only
    `active_fraction * size` nodes participate in events, the rest are
    isolated (zero rows/cols).
    """
    active = max(2, int(size * active_fraction))
    nodes = np.arange(active - 1)
    rows = np.concatenate([nodes, nodes + 1])
    cols = np.concatenate([nodes + 1, nodes])
    data = np.ones(rows.shape[0], dtype=np.float64)

    adjacency = csc_matrix((data, (rows, cols)), shape=(size, size))
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return diags(degrees, format="csc") - adjacency


def synthetic_temporal_laplacian(n_groups=3, n_per_group=20, seed=42):
    """A representative random-walk Laplacian from a SynthTempNetwork run.

    Dogfoods the package: simulates a temporal network (uniformly random
    interactions between grouped individuals), computes the laplacian
    sequence via ContTempNetwork, and returns the laplacian with the most
    non-zeros (busiest time step) as csc.
    """
    from tempnet import ContTempNetwork
    from tempnet.synth_temp_network import Individual, SynthTempNetwork

    # Individual keeps class-level state accumulating across instantiations.
    Individual.all_IDs = []
    Individual.all_groups = []

    np.random.seed(seed)

    inter_tau = 1.0
    activ_tau = 5.0
    individuals = [
        Individual(
            ID=g * n_per_group + i,
            inter_distro_scale=inter_tau,
            activ_distro_scale=activ_tau,
            group=g,
        )
        for g in range(n_groups)
        for i in range(n_per_group)
    ]

    sim = SynthTempNetwork(
        individuals=individuals,
        t_start=0,
        t_end=200.0,
    )
    sim.run()

    network = ContTempNetwork(
        source_nodes=sim.indiv_sources,
        target_nodes=sim.indiv_targets,
        starting_times=sim.start_times,
        ending_times=sim.end_times,
    )
    network.compute_laplacian_matrices()

    busiest = max(network.laplacians, key=lambda L: L.getnnz())
    return busiest.tocsc()
