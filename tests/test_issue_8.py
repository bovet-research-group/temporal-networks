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
    network = ContTempNetwork(
        source_nodes=["A", "B", "A"],
        target_nodes=["B", "C", "B"],
        starting_times=[0, 1, 1],
        ending_times=[3, 2, 4],
        merge_overlapping_events=True,
    )

    assert network._overlapping_events_merged
    assert network.node_to_label_dict == {0: "A", 1: "B", 2: "C"}
    assert network.events_table.shape[0] == 2
