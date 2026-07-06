import pandas as pd
import pytest

from tempnet.temporal_network import ContTempNetwork


def test_merge_overlapping_events_rejects_unrelabelled_string_labels():
    with pytest.raises(ValueError):
        ContTempNetwork(
            source_nodes=["A", "B", "A"],
            target_nodes=["B", "C", "B"],
            starting_times=[0, 1, 1],
            ending_times=[3, 2, 4],
            relabel_nodes=False,
            merge_overlapping_events=True,
        )


def test_merge_overlapping_events_relabels_string_labels_by_default():
    input_events_table = pd.DataFrame({
        "source_nodes": ["A", "B", "A"],
        "target_nodes": ["B", "C", "B"],
        "starting_times": [0, 1, 1],
        "ending_times": [3, 2, 4],
    })

    network = ContTempNetwork(
        events_table=input_events_table,
        merge_overlapping_events=True,
    )

    assert network._overlapping_events_merged
    assert network.node_to_label_dict == {0: "A", 1: "B", 2: "C"}
    assert network.events_table.shape[0] == 2

    reconstructed_events_table = network.events_table[
        ["source_nodes", "target_nodes", "starting_times", "ending_times"]
    ].copy()
    reconstructed_events_table["source_nodes"] = reconstructed_events_table[
        "source_nodes"
    ].map(network.node_to_label_dict)
    reconstructed_events_table["target_nodes"] = reconstructed_events_table[
        "target_nodes"
    ].map(network.node_to_label_dict)

    expected_events_table = pd.DataFrame({
        "source_nodes": ["A", "B"],
        "target_nodes": ["B", "C"],
        "starting_times": [0, 1],
        "ending_times": [4, 2],
    })
    pd.testing.assert_frame_equal(
        reconstructed_events_table.reset_index(drop=True),
        expected_events_table,
        check_dtype=False,
    )
