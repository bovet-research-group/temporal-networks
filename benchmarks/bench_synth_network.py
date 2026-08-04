"""Benchmarks for synthetic temporal-network generation.

Covers the agent-based simulation (``SynthTempNetwork``), the construction
of a ``ContTempNetwork`` from the simulated event lists, and the laplacian
computation on the resulting temporal network.

All benchmarks are seeded so results are comparable across runs and
commits. Benchmarks are meant to be run locally (``asv run``), never in CI.
"""

import numpy as np

from tempnet import ContTempNetwork
from tempnet.synth_temp_network import Individual, SynthTempNetwork


def _make_individuals(n_individuals, n_groups=3):
    """Seeded Individual list; resets accumulating class-level state."""
    Individual.all_IDs = []
    Individual.all_groups = []
    return [
        Individual(
            ID=i,
            inter_distro_scale=1.0,
            activ_distro_scale=5.0,
            group=i % n_groups,
        )
        for i in range(n_individuals)
    ]


class TimeSynthTempNetwork:
    """Simulation cost as a function of population size and duration."""

    params = ([12, 60, 300], [100.0, 500.0])
    param_names = ["n_individuals", "t_end"]
    timeout = 600

    def setup(self, n_individuals, t_end):
        np.random.seed(42)

    def time_run_simulation(self, n_individuals, t_end):
        # construct inside the timed function: run() is not guaranteed to
        # be idempotent across timing repeats (construction cost is
        # negligible against the simulation itself)
        sim = SynthTempNetwork(
            individuals=_make_individuals(n_individuals),
            t_start=0,
            t_end=t_end,
        )
        sim.run()


class TimeContTempNetworkFromSynth:
    """Cost of building/analysing a ContTempNetwork from simulated events."""

    params = ([12, 60, 300],)
    param_names = ["n_individuals"]
    timeout = 600

    def setup(self, n_individuals):
        np.random.seed(42)
        sim = SynthTempNetwork(
            individuals=_make_individuals(n_individuals),
            t_start=0,
            t_end=200.0,
        )
        sim.run()
        self.sim = sim
        self.network = ContTempNetwork(
            source_nodes=sim.indiv_sources,
            target_nodes=sim.indiv_targets,
            starting_times=sim.start_times,
            ending_times=sim.end_times,
        )

    def time_build_network(self, n_individuals):
        ContTempNetwork(
            source_nodes=self.sim.indiv_sources,
            target_nodes=self.sim.indiv_targets,
            starting_times=self.sim.start_times,
            ending_times=self.sim.end_times,
        )

    def time_compute_laplacians(self, n_individuals):
        self.network.compute_laplacian_matrices()
