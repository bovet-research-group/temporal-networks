"""Tests for the Phase 1.5 ``ContTempInstNetwork`` legacy shim.

These tests pin down three guarantees that the upcoming Phase 2 refactor
must not break:

1. Instantiating ``ContTempInstNetwork`` emits a ``DeprecationWarning``.
2. The class is still a subclass of ``ContTempNetwork`` and instances
   register as both.
3. The shim still computes Laplacian matrices end-to-end and they match
   what we get when calling ``compute_laplacian_matrices`` directly on
   the underlying parent loop with the pulse-dynamics hooks (i.e. the
   shim itself does not silently change behaviour relative to Phase 1).
"""

import warnings

import pytest

from tempnet import ContTempInstNetwork, ContTempNetwork
from tempnet._legacy import ContTempInstNetwork as LegacyContTempInstNetwork


def _tiny_kwargs():
    # Two events between three nodes, each one time-unit long so that the
    # pulse and interval semantics produce well-defined, non-empty
    # laplacians.
    return dict(
        source_nodes=[0, 1],
        target_nodes=[1, 2],
        starting_times=[0.0, 1.0],
    )


def test_top_level_and_legacy_export_are_same_class():
    """The package-level re-export must be the same object as the one in
    ``tempnet._legacy``; otherwise pickle round-trips and ``isinstance``
    checks would silently diverge.
    """
    assert ContTempInstNetwork is LegacyContTempInstNetwork


def test_construction_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        ContTempInstNetwork(**_tiny_kwargs())


def test_is_subclass_of_cont_temp_network():
    assert issubclass(ContTempInstNetwork, ContTempNetwork)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        net = ContTempInstNetwork(**_tiny_kwargs())
    assert isinstance(net, ContTempInstNetwork)
    assert isinstance(net, ContTempNetwork)


def test_compute_laplacian_matrices_runs_end_to_end():
    """Smoke test: the shim still produces one laplacian per inter-event
    interval and they have the expected square shape. We do not pin
    numeric values here -- those are covered by the existing
    ``TestContTempInstNetwork`` suite.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        net = ContTempInstNetwork(**_tiny_kwargs())
    net.compute_laplacian_matrices()
    assert hasattr(net, "laplacians")
    assert len(net.laplacians) >= 1
    for L in net.laplacians:
        assert L.shape == (net.num_nodes, net.num_nodes)
