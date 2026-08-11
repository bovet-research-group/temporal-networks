# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmarks for temporal-network constructor sanitization."""

import warnings

import numpy as np
import pandas as pd

from tempnet import ContTempNetwork
from tempnet.sanitize import needs_sanitization

from .common import Machine, pretty_name


machine = Machine()
CONSTRUCTOR_N_EVENTS = machine.get_n_events()

_CONSTRUCTOR_NODES_PER_EVENT = 50  # label diversity per unit of events


def _make_constructor_tables(n_events):
    """Return equivalent unsanitized and normalized event tables.

    Parameters
    ----------
    n_events : int
        Number of events in the generated network.

    Returns
    -------
    tuple of pandas.DataFrame
        The first table requires sanitization. The second table satisfies the
        invariants required by ``sanitize_data=False``.
    """
    rng = np.random.default_rng(42)

    n_nodes = max(2, n_events // _CONSTRUCTOR_NODES_PER_EVENT)
    sources = rng.integers(0, n_nodes, size=n_events)
    targets = (sources + rng.integers(1, n_nodes, size=n_events)) % n_nodes
    starts = rng.uniform(0.0, 1000.0, size=n_events)
    endings = starts + rng.uniform(0.1, 10.0, size=n_events)

    normalized = pd.DataFrame(
        {
            "source_nodes": sources,
            "target_nodes": targets,
            "starting_times": starts,
            "ending_times": endings,
        }
    ).sort_values(
        ["starting_times", "ending_times"],
    ).reset_index(drop=True)

    unsanitized = normalized.copy()
    labels = np.arange(n_nodes) + 1000
    unsanitized["source_nodes"] = labels[
        unsanitized["source_nodes"].to_numpy()
    ]
    unsanitized["target_nodes"] = labels[
        unsanitized["target_nodes"].to_numpy()
    ]
    unsanitized = unsanitized.sample(frac=1.0, random_state=42)
    unsanitized.index = np.arange(10, 10 + len(unsanitized))

    return unsanitized, normalized


class _ConstructorBenchmark:
    """Shared input tables for the constructor benchmarks."""

    params = (CONSTRUCTOR_N_EVENTS,)
    param_names = ["n_events"]
    timeout = 600

    def setup(self, n_events):
        self.unsanitized, self.normalized = _make_constructor_tables(
            n_events
        )
        if not needs_sanitization(self.unsanitized):
            raise RuntimeError(
                "constructor benchmark setup error: unsanitized input"
                " unexpectedly satisfies invariants"
            )
        if needs_sanitization(self.normalized):
            raise RuntimeError(
                "constructor benchmark setup error: normalized input"
                " violates sanitize_data=False invariants"
            )


class TimeContTempNetworkInitialization(_ConstructorBenchmark):
    """Measure constructor time with and without sanitization."""

    @pretty_name("Initialize ContTempNetwork with sanitization")
    def time_sanitized(self, n_events):
        ContTempNetwork(events_table=self.unsanitized)

    @pretty_name("Initialize ContTempNetwork fast track")
    def time_fast_track(self, n_events):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ContTempNetwork(
                events_table=self.normalized,
                sanitize_data=False,
            )


class PeakMemContTempNetworkInitialization(_ConstructorBenchmark):
    """Measure peak constructor memory with and without sanitization."""

    @pretty_name("Peak memory: ContTempNetwork with sanitization")
    def peakmem_sanitized(self, n_events):
        ContTempNetwork(events_table=self.unsanitized)

    @pretty_name("Peak memory: ContTempNetwork fast track")
    def peakmem_fast_track(self, n_events):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ContTempNetwork(
                events_table=self.normalized,
                sanitize_data=False,
            )
