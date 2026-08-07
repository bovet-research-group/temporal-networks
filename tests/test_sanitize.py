"""
#
# Temporal networks `tempnet`
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@uzh.ch>
# Copyright (C) 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>, 
#                    Yasaman Asgari <yasaman.asgari@uzh.ch>, 
#                    Samuel Koovely <samuel.koovely@uzh.ch>, 
#                    Jonas Liechti <jonas@t4d.ch>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""

"""Tests for tempnet.sanitize: event-table normalization helpers."""

import numpy as np
import pandas as pd
import pytest

from tempnet.sanitize import (
    SanitizedTable,
    _build_label_maps,
    _is_contiguous,
    needs_sanitization,
    sanitize_events_table,
)


def make_df(sources, targets, starts, ends=None, index=None):
    data = {
        "source_nodes": sources,
        "target_nodes": targets,
        "starting_times": starts,
    }
    if ends is not None:
        data["ending_times"] = ends
    df = pd.DataFrame(data)
    if index is not None:
        df.index = index
    return df


@pytest.fixture
def normalized_df():
    """A table that already satisfies all invariants."""
    return make_df(
        sources=[0, 1, 2], targets=[1, 2, 0],
        starts=[0.0, 1.0, 2.0], ends=[1.0, 2.0, 3.0],
    )


# --------------------------------------------------------------------------- #
# _is_contiguous
# --------------------------------------------------------------------------- #
class TestIsContiguous:

    def test_contiguous_from_zero(self):
        assert _is_contiguous([0, 1, 2], [1, 2, 0]) is True

    def test_contiguous_split_over_src_tgt(self):
        # neither column alone covers 0..N-1, together they do
        assert _is_contiguous([0, 2], [1, 3]) is True

    def test_not_starting_at_zero(self):
        assert _is_contiguous([1, 2, 3], [2, 3, 1]) is False

    def test_gap_in_ids(self):
        assert _is_contiguous([0, 1, 3], [1, 3, 0]) is False

    def test_string_labels(self):
        assert _is_contiguous(["a", "b"], ["b", "a"]) is False

    def test_float_ids(self):
        assert _is_contiguous([0.0, 1.0], [1.0, 0.0]) is False

    def test_empty(self):
        assert _is_contiguous([], []) is True


# --------------------------------------------------------------------------- #
# _build_label_maps
# --------------------------------------------------------------------------- #
class TestBuildLabelMaps:

    def test_sorted_labels_get_ascending_ids(self):
        l2n, n2l = _build_label_maps([30, 10], [20, 10])
        assert l2n == {10: 0, 20: 1, 30: 2}
        assert n2l == {0: 10, 1: 20, 2: 30}

    def test_maps_are_inverse(self):
        l2n, n2l = _build_label_maps(["b", "a"], ["c", "b"])
        for label, node in l2n.items():
            assert n2l[node] == label


# --------------------------------------------------------------------------- #
# needs_sanitization
# --------------------------------------------------------------------------- #
class TestNeedsSanitization:

    def test_normalized_table_is_clean(self, normalized_df):
        assert needs_sanitization(normalized_df) is False

    def test_non_contiguous_ids(self):
        df = make_df([10, 20, 30], [20, 30, 10],
                     starts=[0.0, 1.0, 2.0], ends=[1.0, 2.0, 3.0])
        assert needs_sanitization(df) is True

    def test_string_labels(self):
        df = make_df(["a", "b"], ["b", "a"],
                     starts=[0.0, 1.0], ends=[1.0, 2.0])
        assert needs_sanitization(df) is True

    def test_unsorted_starting_times(self):
        df = make_df([0, 1], [1, 0], starts=[2.0, 0.0], ends=[3.0, 1.0])
        assert needs_sanitization(df) is True

    def test_unsorted_ending_times_tiebreak(self):
        # equal starts, decreasing ends: violates the secondary sort key
        df = make_df([0, 1], [1, 0], starts=[0.0, 0.0], ends=[3.0, 1.0])
        assert needs_sanitization(df) is True

    def test_non_range_index(self, normalized_df):
        df = normalized_df.copy()
        df.index = [10, 20, 30]
        assert needs_sanitization(df) is True

    def test_duplicate_index(self):
        df = pd.concat([
            make_df([0], [1], starts=[0.0], ends=[1.0]),
            make_df([1], [2], starts=[2.0], ends=[3.0]),
        ])  # index [0, 0]
        # ids [0,1,2] are contiguous and events sorted, but the index is bad
        assert needs_sanitization(df) is True

    def test_named_index(self, normalized_df):
        df = normalized_df.copy()
        df.index.name = "event_id"
        assert needs_sanitization(df) is True

    def test_without_ending_times_column(self):
        # instantaneous-style table: only starting_times used for the check
        df = make_df([0, 1], [1, 0], starts=[0.0, 1.0])
        assert needs_sanitization(df) is False

    def test_empty_table(self):
        df = make_df([], [], starts=[])
        assert needs_sanitization(df) is False


# --------------------------------------------------------------------------- #
# sanitize_events_table
# --------------------------------------------------------------------------- #
class TestSanitizeEventsTable:

    def test_returns_named_tuple(self, normalized_df):
        result = sanitize_events_table(normalized_df)
        assert isinstance(result, SanitizedTable)

    def test_normalized_input_is_unchanged(self, normalized_df):
        result = sanitize_events_table(normalized_df)
        pd.testing.assert_frame_equal(result.events_table, normalized_df)
        assert result.label_to_node_dict == {0: 0, 1: 1, 2: 2}
        assert result.node_to_label_dict == {0: 0, 1: 1, 2: 2}

    def test_output_never_needs_sanitization(self):
        df = make_df([30, 10, 20], [10, 20, 30],
                     starts=[2.0, 0.0, 1.0], ends=[3.0, 1.0, 2.0],
                     index=[7, 5, 6])
        result = sanitize_events_table(df)
        assert needs_sanitization(result.events_table) is False

    def test_relabels_non_contiguous_ids(self):
        df = make_df([10, 20, 30], [20, 30, 10],
                     starts=[0.0, 1.0, 2.0], ends=[1.0, 2.0, 3.0])
        result = sanitize_events_table(df)
        used = set(result.events_table.source_nodes) | \
            set(result.events_table.target_nodes)
        assert used == {0, 1, 2}
        assert result.label_to_node_dict == {10: 0, 20: 1, 30: 2}
        assert result.node_to_label_dict == {0: 10, 1: 20, 2: 30}

    def test_sorts_chronologically_and_resets_index(self):
        df = make_df([0, 1, 2], [1, 2, 0],
                     starts=[2.0, 0.0, 1.0], ends=[3.0, 1.0, 2.0],
                     index=[20, 10, 30])
        result = sanitize_events_table(df)
        assert result.events_table.starting_times.tolist() == [0.0, 1.0, 2.0]
        assert result.events_table.index.tolist() == [0, 1, 2]

    def test_sort_uses_ending_times_as_tiebreak(self):
        df = make_df([0, 1], [1, 0], starts=[0.0, 0.0], ends=[3.0, 1.0])
        result = sanitize_events_table(df)
        assert result.events_table.ending_times.tolist() == [1.0, 3.0]

    def test_clears_index_name(self, normalized_df):
        df = normalized_df.copy()
        df.index.name = "event_id"
        result = sanitize_events_table(df)
        assert result.events_table.index.name is None

    def test_copy_by_default(self):
        df = make_df([10, 20], [20, 10], starts=[1.0, 0.0], ends=[2.0, 1.0])
        df_before = df.copy()
        result = sanitize_events_table(df)
        assert result.events_table is not df
        pd.testing.assert_frame_equal(df, df_before)  # caller untouched

    def test_inplace_modifies_input(self):
        df = make_df([10, 20], [20, 10], starts=[1.0, 0.0], ends=[2.0, 1.0])
        result = sanitize_events_table(df, inplace=True)
        assert result.events_table is df
        assert df.source_nodes.tolist() == [1, 0]  # relabelled in place
        assert df.starting_times.tolist() == [0.0, 1.0]  # sorted in place

    def test_user_label_map_applied(self):
        df = make_df(["x", "y"], ["y", "x"], starts=[0.0, 1.0],
                     ends=[1.0, 2.0])
        result = sanitize_events_table(
            df, label_to_node_dict={"x": 0, "y": 1},
        )
        np.testing.assert_array_equal(
            result.events_table.source_nodes.values, [0, 1])
        np.testing.assert_array_equal(
            result.events_table.target_nodes.values, [1, 0])
        assert result.node_to_label_dict == {0: "x", 1: "y"}

    def test_user_label_map_non_unique_raises(self):
        df = make_df(["x", "y"], ["y", "x"], starts=[0.0, 1.0],
                     ends=[1.0, 2.0])
        with pytest.raises(ValueError):
            sanitize_events_table(df, label_to_node_dict={"x": 0, "y": 0})

    def test_user_label_map_non_canonical_raises(self):
        df = make_df(["x", "y"], ["y", "x"], starts=[0.0, 1.0],
                     ends=[1.0, 2.0])
        with pytest.raises(ValueError):
            sanitize_events_table(df, label_to_node_dict={"x": 0, "y": 2})

    def test_without_ending_times_column(self):
        # instantaneous-style table: sortable without ending_times
        df = make_df([1, 2], [2, 1], starts=[1.0, 0.0])
        result = sanitize_events_table(df)
        assert result.events_table.starting_times.tolist() == [0.0, 1.0]
        used = set(result.events_table.source_nodes) | \
            set(result.events_table.target_nodes)
        assert used == {0, 1}

    def test_empty_table(self):
        df = make_df([], [], starts=[], ends=[])
        result = sanitize_events_table(df)
        assert len(result.events_table) == 0
        assert result.label_to_node_dict == {}

    def test_extra_columns_preserved_and_aligned(self):
        df = make_df([0, 1], [1, 0], starts=[2.0, 0.0], ends=[3.0, 1.0])
        df["weight"] = [20.0, 0.0]  # same order as starts
        result = sanitize_events_table(df)
        # after sorting, weights must follow their events
        assert result.events_table.weight.tolist() == [0.0, 20.0]
