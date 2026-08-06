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


import os
import pickle
import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

from tempnet.utils import to_dense

from tempnet.temporal_network import ContTempNetwork, ContTempInstNetwork


MICE_URL = (
    "https://zenodo.org/record/4725155/files/mice_contact_sequence.csv.gz"
)


# HELPERS 
def make_df(sources, targets, starts=None, ends=None):
    """Build an events_table DataFrame"""
    data = {
        "source_nodes": sources,
        "target_nodes": targets,
        "starting_times": starts,
    }
    if ends is not None:
        data["ending_times"] = ends
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_network():
    """3-node, 4-event network whose properties are verified by hand.

    Events (source, target, start, end):
        0: (A, B, 0, 2)
        1: (B, C, 1, 3)
        2: (A, C, 4, 5)
        3: (A, B, 6, 7)
    """
    return ContTempNetwork(
        source_nodes=["A", "B", "A", "A"],
        target_nodes=["B", "C", "C", "B"],
        starting_times=[0, 1, 4, 6],
        ending_times=[2, 3, 5, 7])

@pytest.fixture
def network_overlapping():
    """Two overlapping events on the same node pair"""
    return ContTempNetwork(
        source_nodes=["A", "B", "A"],
        target_nodes=["B", "C", "B"],
        starting_times=[0, 1, 1],
        ending_times=[3, 2, 4],
        merge_overlapping_events=True,
    )

@pytest.fixture
def prepared_network(simple_network):
    """simple_network with laplacians computed, ready for T computation."""
    simple_network.compute_laplacian_matrices()
    return simple_network


@pytest.fixture
def mice_network():
    """Load the first 24h of the mice contact dataset from Zenodo."""
    cut_after = 24 * 3600
    raw_df = pd.read_csv(MICE_URL, compression="gzip")
    raw_df = raw_df[raw_df["ending_times"] < cut_after]
    return ContTempNetwork(
        source_nodes=raw_df["source_nodes"].tolist(),
        target_nodes=raw_df["target_nodes"].tolist(),
        starting_times=raw_df["starting_times"].round(3).tolist(),
        ending_times=raw_df["ending_times"].round(3).tolist(),
        relabel_nodes=True,
    )


@pytest.fixture
def simple_unsorted_network():
    """Network namespace with simple events in non-chronological order.

    The fixture is used to expose constructor differences between list input
    and DataFrame input. The list constructor sorts events by
    ``starting_times``/``ending_times`` and resets the index, while the
    DataFrame constructor historically preserves caller order and index.
    """
    network = SimpleNamespace()
    network.source_nodes = [1, 4, 2, 7, 5, 3, 9, 6, 8, 10]
    network.target_nodes = [2, 5, 3, 8, 6, 4, 10, 7, 9, 1]
    network.starting_times = [0, 2, 0.5, 5, 4, 1, 5, 4, 5, 3]
    network.ending_times = [3, 7, 1, 6, 5, 2, 6, 6, 7, 4]
    network.events_table = pd.DataFrame({
        "source_nodes": network.source_nodes,
        "target_nodes": network.target_nodes,
        "starting_times": network.starting_times,
        "ending_times": network.ending_times,
    })
    nodes = set()
    nodes.update(network.source_nodes)
    nodes.update(network.target_nodes)
    network.nodes = sorted(nodes)
    network.node_label_id_map = {
        node: _id for _id, node in enumerate(network.nodes)
    }
    return network


class TestSimpleNetwork: 

    def test_num_nodes(self, simple_network):
        assert simple_network.num_nodes == 3

    def test_num_events(self, simple_network):
        assert simple_network.num_events == 4

    def test_start_time(self, simple_network):
        assert simple_network.start_time == 0

    def test_end_time(self, simple_network):
        assert simple_network.end_time == 7

    def test_print(self ,simple_network):
        s=str(simple_network)
        assert s==  "<class 'tempnet.temporal_network.ContTempNetwork'> with 3 nodes and 4 events" 

    def test_node_array_sorted(self, simple_network):
        assert list(simple_network.node_array) == [0, 1, 2]

    def test_durations_column_exists(self, simple_network):
        assert "durations" in simple_network.events_table.columns

    def test_durations_values(self, simple_network):
        assert list(simple_network.events_table["durations"]) == [2, 2, 1, 1]

    def test_events_sorted_by_start(self, simple_network):
        starts = simple_network.events_table["starting_times"].tolist()
        assert starts == sorted(starts)

    def test_required_columns_present(self, simple_network):
        for col in ["source_nodes", "target_nodes", "starting_times",
                    "ending_times", "durations"]:
            assert col in simple_network.events_table.columns

    def test_active_events(self, simple_network):
        assert simple_network.num_active_events(t_start=None, t_end=None)==4
        assert simple_network.num_active_events(t_start=1, t_end=2)==2
        assert simple_network.num_active_events(t_start=None, t_end=5)==3
        assert simple_network.num_active_events(t_start=6.5, t_end=None)==1
        assert simple_network.num_active_events(t_start=3.5, t_end=3.75)==0

        with pytest.raises(ValueError):
            simple_network.num_active_events(t_start=5, t_end=3)

        with pytest.raises(ValueError):
            simple_network.num_active_events(t_start=1, t_end=1)

    def test_active_nodes(self, simple_network):
        assert simple_network.num_active_nodes(t_start=None, t_end=None)==3
        assert simple_network.num_active_nodes(t_start=1, t_end=2)==3
        assert simple_network.num_active_nodes(t_start=None, t_end=5)==3
        assert simple_network.num_active_nodes(t_start=6.5, t_end=None)==2
        assert simple_network.num_active_nodes(t_start=3.5, t_end=3.75)==0

        with pytest.raises(ValueError):
            simple_network.num_active_nodes(t_start=5, t_end=3)

        with pytest.raises(ValueError):
            simple_network.num_active_nodes(t_start=1, t_end=1)

    def test_active_events_boundary_overlap_semantics(self, simple_network):
        """Events are active when they overlap with positive duration."""
        # Window [2, 4): event ending at 2 is excluded, event starting at 4
        # is excluded; only [1, 3) overlaps.
        assert simple_network.num_active_events(t_start=2, t_end=4) == 1

        # Window [4, 5): event starting exactly at t_start is included.
        assert simple_network.num_active_events(t_start=4, t_end=5) == 1

        # Window [3, 5): event ending at t_start is excluded, event ending
        # at t_end is included because it overlaps before t_end.
        assert simple_network.num_active_events(t_start=3, t_end=5) == 1

        # Window [5, 6): event ending at t_start and event starting at t_end
        # are both excluded.
        assert simple_network.num_active_events(t_start=5, t_end=6) == 0

    def test_active_nodes_boundary_overlap_semantics(self, simple_network):
        """Active nodes follow the same overlap convention as events."""
        assert simple_network.active_nodes(t_start=2,
                                           t_end=4).tolist() == [1, 2]
        assert simple_network.num_active_nodes(t_start=2, t_end=4) == 2

        assert simple_network.active_nodes(t_start=4,
                                           t_end=5).tolist() == [0, 2]
        assert simple_network.num_active_nodes(t_start=4, t_end=5) == 2

        assert simple_network.active_nodes(t_start=3,
                                           t_end=5).tolist() == [0, 2]
        assert simple_network.num_active_nodes(t_start=3, t_end=5) == 2

        assert simple_network.active_nodes(t_start=5, t_end=6).tolist() == []
        assert simple_network.num_active_nodes(t_start=5, t_end=6) == 0

    def test_adj_full(self, simple_network):
        A = simple_network.compute_static_adjacency_matrix().toarray()
        expected = np.array([
            [0, 3, 1],
            [3, 0, 2],
            [1, 2, 0],
        ])
        assert np.allclose(A, expected)

    def test_adj_window_0_2(self, simple_network):
        A = simple_network.compute_static_adjacency_matrix(
            start_time=0, end_time=2,
        ).toarray()
        expected = np.array([
            [0, 2, 0],
            [2, 0, 1],
            [0, 1, 0],
        ])
        assert np.allclose(A, expected)

    def test_adj_window_2p5_3(self, simple_network):
        A = simple_network.compute_static_adjacency_matrix(
            start_time=2.5, end_time=3,
        ).toarray()
        expected = np.array([
            [0, 0, 0],
            [0, 0, 0.5],
            [0, 0.5, 0],
        ])
        assert np.allclose(A, expected)

    @pytest.mark.parametrize("dynamics", [ "rw", "heat"])
    def test_laplacians_count(self, simple_network, dynamics):
        """One Laplacian per inter-event step over the full grid.

        times = [0,1,2,3,4,5,6,7] -> 7 inter-event steps.
        """
        simple_network.compute_laplacian_matrices(dynamics=dynamics)
        assert len(simple_network.laplacians) == 7

    def test_laplacian_step_1_2_randomwalk(self, simple_network):
        """Step [1,2]: A-B and B-C active. Random-walk Laplacian I - D^-1 A."""
        simple_network.compute_laplacian_matrices(dynamics="rw")
        L = simple_network.laplacians[1].toarray()
        expected = np.array([
            [1.0, -1.0, 0.0],
            [-0.5, 1.0, -0.5],
            [0.0, -1.0, 1.0],
        ])
        assert np.allclose(L, expected)


    def test_laplacian_step_1_2_heat(self, simple_network):
        """Step [1,2]: A-B and B-C active. Heat kernel Laplacian."""
        simple_network.compute_laplacian_matrices(dynamics="heat")
        L = simple_network.laplacians[1].toarray()
        expected = np.array([
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ])
        assert np.allclose(L, expected)


    def test_laplacian_step_0_1_connected_block(self, simple_network):
        """Step [0,1]: only A-B active; C isolated. C's diagonal is
        convention-dependent, so only the A-B block and zero coupling to C
        are asserted.
        """
        simple_network.compute_laplacian_matrices()
        L = simple_network.laplacians[0].toarray()
        assert L[0, 0] == 1
        assert L[0, 1] == -1
        assert L[1, 0] == -1
        assert L[1, 1] == 1
        assert L[0, 2] == 0
        assert L[1, 2] == 0
        assert L[2, 0] == 0
        assert L[2, 1] == 0
        assert L[2, 2] == 0  # self loop

    @pytest.mark.parametrize("dynamics", [ "rw", "heat"])
    def test_laplacian_empty_step_all_zero(self, simple_network, dynamics):
        """Step [3,4]: no active events, so the Laplacian is all zeros."""
        simple_network.compute_laplacian_matrices(dynamics=dynamics)
        L = simple_network.laplacians[3].toarray()
        assert np.all(L == 0)

    @pytest.mark.parametrize("dynamics", [ "rw", "heat"])
    def test_laplacian_rows_sum_zero_connected_step(self, simple_network, dynamics):
        simple_network.compute_laplacian_matrices(dynamics=dynamics)
        n = simple_network.num_nodes
        for i in range(len(simple_network.laplacians)):
            L = simple_network.laplacians[i].toarray()
            assert L.shape == (n, n)
            assert np.allclose(L.sum(axis=1), 0.0)


class TestTempNetwork:
    def setup_method(self):
        # ###
        # folder setup
        self.temp_dir = tempfile.gettempdir()
        self.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
        # ###
        # Test data
        # create a minimmal network
        self.minimal = SimpleNamespace()
        self.minimal.source_nodes = [1, 2, 3, 4, 5]
        self.minimal.target_nodes = [2, 3, 4, 5, 1]
        self.minimal.starting_times = [0.5, 1.0, 2.0, 2.0, 3.0]
        self.minimal.ending_times = [1.5, 1.5, 2.5, 4.0, 4.0]
        self.minimal.extra_attrs = {"attr1": [True, False]}
        self.minimal.events_table = self._to_df(self.minimal)
        self.minimal.nodes = self._get_nodes(self.minimal)
        self.minimal.node_label_id_map = self._get_label_id_map(self.minimal)
        self.minimal.tmp_pkl = tempfile.NamedTemporaryFile(suffix='.pkl',
                                                           delete=False)
        self.minimal.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                            delete=False)
        self.minimal_instant = copy(self.minimal)
        del self.minimal_instant.ending_times
        self.minimal_instant.events_table = self.minimal.events_table.drop(
            ContTempNetwork._ENDINGS, axis=1
        )

        # create a simple network
        self.simple = SimpleNamespace()
        # we assume 10 nodes, and each starting a connection in order
        self.simple.source_nodes = list(range(1, 11))
        # target nodes are also in order
        self.simple.target_nodes = list(range(2, 11)) + [1]
        self.simple.starting_times = [0, 0.5, 1, 2, 3, 4, 4, 5, 5, 5]
        self.simple.ending_times = [3, 1, 2, 7, 4, 5, 6, 6, 6, 7]
        self.simple.events_table = self._to_df(self.simple)
        self.simple.nodes = self._get_nodes(self.simple)
        self.simple.node_label_id_map = self._get_label_id_map(self.simple)
        self.simple.tmp_pkl = tempfile.NamedTemporaryFile(suffix='.pkl',
                                                          delete=False)
        self.simple.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                           delete=False)
        self.simple_instant = copy(self.simple)
        del self.simple_instant.ending_times
        self.simple_instant.events_table = self.simple.events_table.drop(
            ContTempNetwork._ENDINGS, axis=1
        )

        # Load real data
        self.real = SimpleNamespace()
        self.real.url = 'https://zenodo.org/record/4725155/files/'\
                        'mice_contact_sequence.csv.gz'
        self.real.raw_df = pd.read_csv(self.real.url,
                                       compression='gzip')
        self.real.cut_after = 3600  # only use first hour
        self.real.events_table = self.real.raw_df[
            self.real.raw_df['ending_times'] < 3600
        ]
        # self.real.source_nodes = self.real.raw_df['source_nodes'].tolist()
        # self.real.target_nodes = self.real.raw_df['target_nodes'].tolist()
        # self.real.starting_times = self.real.raw_df['starting_times'].tolist()
        # self.real.ending_times = self.real.raw_df['ending_times'].tolist()

        # ###
        # gather all networks
        self.networks = [self.minimal, self.minimal_instant,
                         self.simple, self.simple_instant]
        self.minimals = [self.minimal, self.minimal_instant]

    @staticmethod
    def _to_df(network: SimpleNamespace):
        """Convert a network from a namespace ot a data frame
        """
        as_df = pd.DataFrame({
            "source_nodes": network.source_nodes,
            "target_nodes": network.target_nodes,
            "starting_times": network.starting_times,
        })
        ending_times = getattr(network, ContTempNetwork._ENDINGS, None)
        if ending_times is not None:
            as_df[ContTempNetwork._ENDINGS] = ending_times
        return as_df

    @staticmethod
    def _get_instance(network: SimpleNamespace, use_df=False, **params):
        is_instant = not hasattr(network, ContTempNetwork._ENDINGS)
        cls = ContTempInstNetwork if is_instant else ContTempNetwork
        if use_df:
            return cls(events_table=network.events_table, **params)
        else:
            kwargs = dict(
                source_nodes=network.source_nodes,
                target_nodes=network.target_nodes,
                starting_times=network.starting_times,
                **params,
            )
            if not is_instant:
                kwargs["ending_times"] = network.ending_times
            return cls(**kwargs)

    @staticmethod
    def _get_nodes(network: SimpleNamespace):
        """Get a sorted list of nodes
        """
        nodes = set()
        nodes.update(network.source_nodes)
        nodes.update(network.target_nodes)
        return sorted(nodes)

    @staticmethod
    def _get_label_id_map(network: SimpleNamespace):
        """Get the mapping from node labels to internal ID
        """
        return {node: _id for _id, node in enumerate(network.nodes)}

    def teardown_method(self):
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if 'temp.pkl' in file or 'temp.json' in file:
                os.remove(os.path.join(temp_dir, file))

    def test_init_with_source_and_target_nodes(self):
        for network in self.networks:
            temp_network = self._get_instance(network, use_df=False)
            assert isinstance(temp_network, ContTempNetwork)

    def test_init_without_source_nodes(self):
        for network in self.networks:
            with pytest.raises(AssertionError):
                ContTempNetwork(target_nodes=network.target_nodes)

    def test_init_without_target_nodes(self):
        for network in self.networks:
            with pytest.raises(AssertionError):
                ContTempNetwork(source_nodes=network.source_nodes)

    def test_init_inconsistent_event_numbers_type(self):
        """Make sure we detect variable numbers of events
        """
        with pytest.raises(AssertionError):
            # error in source and target nodes
            ContTempNetwork(source_nodes=[1, 2, 3], target_nodes=[1, 2],
                            starting_times=[0, 0], ending_times=[1, 1])
            # not enough ending times
            ContTempNetwork(source_nodes=[1, 2], target_nodes=[1, 2],
                            starting_times=[0, 0], ending_times=[1])

    def test_init_missing_params(self):
        """Make sure we detect variable numbers of events
        """
        with pytest.raises(AssertionError):
            # missing starting times
            ContTempNetwork(source_nodes=[1, 2], target_nodes=[1, 2],)

    def test_init_with_inconsistent_node_type(self):
        with pytest.raises(TypeError):
            # int and str cannot be compared > type error
            ContTempNetwork(source_nodes=[0, 1], target_nodes=['a', 'b'],
                            starting_times=[0, 0], ending_times=[1, 1])

    @pytest.fixture
    def saved_network(self):
        def _get_network(network: SimpleNamespace, use_df=False, **params):
            temp_network = self._get_instance(network, use_df=use_df, **params)
            with open(network.tmp_pkl.name, 'wb') as f:
                pickle.dump(temp_network, f)
                return temp_network
        return _get_network

    @pytest.fixture
    def get_loaded_network(self):
        def loaded_network(network):
            with open(network.tmp_pkl.name, 'rb') as f:
                return pickle.load(f)
        return loaded_network

    def test_save_and_load_pickle(self, saved_network, get_loaded_network):
        for network in self.networks:
            temp_network = saved_network(network=network, use_df=True)
            assert isinstance(temp_network, ContTempNetwork)

            # first save is again
            temp_network = saved_network(network=network, use_df=False)
            # load the temp network
            loaded_network = get_loaded_network(network=network)
            assert isinstance(loaded_network, ContTempNetwork)
            sn_et = temp_network.events_table
            ln_et = loaded_network.events_table
            pd.testing.assert_series_equal(sn_et.source_nodes,
                                           ln_et.source_nodes)
            pd.testing.assert_series_equal(sn_et.target_nodes,
                                           ln_et.target_nodes)
            pd.testing.assert_series_equal(sn_et.starting_times,
                                           ln_et.starting_times)
            pd.testing.assert_series_equal(sn_et.ending_times,
                                           ln_et.ending_times)

    def test_save_and_load_preserves_laplacian_dynamics(self):
        network = self._get_instance(self.simple, use_df=True)
        network.compute_laplacian_matrices(dynamics="heat")

        network.save(self.simple.tmp_pkl.name)
        loaded_network = ContTempNetwork.load(self.simple.tmp_pkl.name)

        assert loaded_network.laplacian_dynamics == "heat"

    def test_import_data(self):
        """Make sure we can work with data with incomplete node lists
        """
        network = ContTempNetwork(
            events_table=self.real.events_table,
            merge_overlapping_events=False
        )
        network.compute_laplacian_matrices()

    def test_merge_overlapping_events(self):
        # create a network with overlapping events
        source_nodes = [0, 0]
        target_nodes = [1, 2]
        starting_times = [0.5, 1.0]
        ending_times = [1.0, 1.5]
        extra_attrs = {"attr1": [True, False]}
        events_table = pd.DataFrame({
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "starting_times": starting_times,
            "ending_times": ending_times
        })
        network = ContTempNetwork(events_table=events_table,
                                  merge_overlapping_events=True)
        assert network._overlapping_events_merged

    def test_time_grid(self):
        for network in self.networks:
            temp_network = self._get_instance(network, use_df=True)
            temp_network._compute_time_grid()

    def test_unsorted_list_input_is_sorted_and_index_reset(
        self,
        simple_unsorted_network,
    ):
        """List input is normalized to chronological order with a RangeIndex.

        This documents the existing list-constructor behavior. The final
        laplacian computation is a smoke test that the reset labels remain
        compatible with later ``.loc``-based event lookups.
        """
        network = self._get_instance(simple_unsorted_network, use_df=False)

        assert network.events_table.starting_times.tolist() == sorted(
            simple_unsorted_network.starting_times
        )
        assert network.events_table.index.tolist() == list(
            range(network.num_events)
        )

        network.compute_laplacian_matrices()
        assert len(network.laplacians) > 0

    def test_unsorted_dataframe_matches_list_constructor_order(
        self,
        simple_unsorted_network,
    ):
        """DataFrame input should match list input for the same event records.

        Regression test: list input sorts events chronologically, while
        DataFrame input has historically preserved caller order. Downstream
        code assumes chronological event order in several places, so both
        constructor paths should normalize to the same internal event table.
        """
        net_lists = self._get_instance(simple_unsorted_network, use_df=False)
        net_df = self._get_instance(simple_unsorted_network, use_df=True)

        cols = ["source_nodes", "target_nodes",
                "starting_times", "ending_times"]
        pd.testing.assert_frame_equal(
            net_lists.events_table[cols].reset_index(drop=True),
            net_df.events_table[cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_dataframe_fast_path_preserves_order_and_index(self):
        """`relabel_nodes=False` keeps the caller-provided table unchanged."""
        events_table = pd.DataFrame({
            "source_nodes": [0, 1, 2],
            "target_nodes": [1, 2, 0],
            "starting_times": [0.0, 1.0, 2.0],
            "ending_times": [1.0, 2.0, 3.0],
        })

        network = ContTempNetwork(
            events_table=events_table,
            relabel_nodes=False,
            node_to_label_dict={0: 0, 1: 1, 2: 2},
        )

        assert network.events_table is events_table
        pd.testing.assert_frame_equal(network.events_table, events_table)

    def test_csv_fast_path_preserves_order_and_index(self, tmp_path):
        """`relabel_nodes=False` does not reset CSV-loaded event tables."""
        events_table = pd.DataFrame({
            "source_nodes": [0, 1, 2],
            "target_nodes": [1, 2, 0],
            "starting_times": [2.0, 0.0, 1.0],
            "ending_times": [3.0, 1.0, 2.0],
        }, index=[20, 10, 30])
        csv_path = tmp_path / "events.csv"
        events_table.to_csv(csv_path)

        network = ContTempNetwork(
            events_table=csv_path,
            relabel_nodes=False,
            node_to_label_dict={0: 0, 1: 1, 2: 2},
            index_col=0,
        )

        assert network.events_table.index.tolist() == [20, 10, 30]
        assert network.events_table.starting_times.tolist() == [2.0, 0.0, 1.0]

    def test_inst_events_table_matches_start_plus_one_interval(self):
        """ContTempInstNetwork synthesizes ending_times = start + 1.

        The resulting events_table must equal that of an interval
        ContTempNetwork explicitly constructed with the same
        ending_times.

        Note: laplacian equality is intentionally not asserted here.
        ContTempInstNetwork.compute_laplacian_matrices implements pulse
        dynamics (state reset every step, no-op on event end), matching
        upstream TemporalNetwork.py at commit f99bca3, which is
        fundamentally distinct from the parent's interval dynamics.
        """
        starts = self.minimal.starting_times
        interval = ContTempNetwork(
            source_nodes=self.minimal.source_nodes,
            target_nodes=self.minimal.target_nodes,
            starting_times=starts,
            ending_times=[s + 1 for s in starts],
        )
        inst = ContTempInstNetwork(
            source_nodes=self.minimal.source_nodes,
            target_nodes=self.minimal.target_nodes,
            starting_times=starts,
        )
        pd.testing.assert_frame_equal(
            interval.events_table.reset_index(drop=True),
            inst.events_table.reset_index(drop=True),
        )

# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #
class TestConstructorValidation:

    def test_init_with_source_and_target_nodes(self, simple_network):
        assert isinstance(simple_network, ContTempNetwork)

    def test_init_without_source_nodes(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(target_nodes=[1, 2])

    def test_init_without_target_nodes(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(source_nodes=[1, 2])

    def test_init_missing_starting_times(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(source_nodes=[1, 2], target_nodes=[1, 2])

    def test_mismatched_source_target_lengths(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(
                source_nodes=[1, 2, 3], target_nodes=[1, 2],
                starting_times=[0, 0], ending_times=[1, 1],
            )

    def test_mismatched_ending_times_length(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(
                source_nodes=[1, 2], target_nodes=[1, 2],
                starting_times=[0, 0], ending_times=[1],
            )

    def test_inconsistent_node_type_raises(self):
        # int and str cannot be compared -> TypeError while sorting nodes
        with pytest.raises(TypeError):
            ContTempNetwork(
                source_nodes=[0, 1], target_nodes=["a", "b"],
                starting_times=[0, 0], ending_times=[1, 1],
            )
    def test_wrong_file(self):
        with pytest.raises(ValueError):
            ContTempNetwork(events_table="not_a_file.csv")
    
    def test_empty_dataframe(self,tmp_path):
        with open(tmp_path/"empty.csv", "w") as f:
            f.write("\n\n")
        with pytest.raises(ValueError):
            ContTempNetwork(events_table="empty.csv")

    def test_missing_required_columns(self,tmp_path):
        df = pd.DataFrame({
            "source_nodes": [0, 1],
            "target_nodes": [1, 0],
        })
        df.to_csv(tmp_path/"temp_missing_columns.csv", index=False)
        with pytest.raises(ValueError):
            ContTempNetwork(events_table="temp_missing_columns.csv")

    def test_extra_attrs_wrong_length_raises(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(
                source_nodes=["A"], target_nodes=["B"],
                starting_times=[0], ending_times=[1],
                extra_attrs={"weight": [1.0, 2.0]},  # too long
            )
    def test_extra_attrs_correct_length(self):
            ContTempNetwork(
                source_nodes=["A"], target_nodes=["B"],
                starting_times=[0], ending_times=[1],
                extra_attrs={"sex_source": [1.0], "sex_target": [0.0]},  
            )
        

    def test_invalid_events_table_type_raises(self):
        with pytest.raises(ValueError):
            ContTempNetwork(events_table=12345)

    def test_events_table_missing_required_column_raises(self):
        with pytest.raises(ValueError):
            ContTempNetwork(events_table=pd.DataFrame({"source_nodes": [0, 1]}))

    def test_compute_time_grid(self, simple_network):
        simple_network._compute_time_grid()


    @pytest.fixture
    def get_loaded_network(self):
        def loaded_network(network):
            with open(network.tmp_pkl.name, 'rb') as f:
                return pickle.load(f)
        return loaded_network

    def test_import_data(self):
        """Make sure we can work with data with incomplete node lists
        """
        network = ContTempNetwork(
            events_table=self.real.events_table,
            merge_overlapping_events=False
        )
        network.compute_laplacian_matrices()

    def test_merge_overlapping_events(self):
        # create a network with overlapping events
        source_nodes = [0, 0]
        target_nodes = [1, 2]
        starting_times = [0.5, 1.0]
        ending_times = [1.0, 1.5]
        extra_attrs = {"attr1": [True, False]}
        events_table = pd.DataFrame({
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "starting_times": starting_times,
            "ending_times": ending_times
        })
        network = ContTempNetwork(events_table=events_table,
                                  merge_overlapping_events=True)
        assert network._overlapping_events_merged

    def test_dataframe_fast_path_preserves_order_and_index(self):
        """`relabel_nodes=False` keeps the caller-provided table unchanged."""
        events_table = pd.DataFrame({
            "source_nodes": [0, 1, 2],
            "target_nodes": [1, 2, 0],
            "starting_times": [0.0, 1.0, 2.0],
            "ending_times": [1.0, 2.0, 3.0],
        })

        network = ContTempNetwork(
            events_table=events_table,
            relabel_nodes=False,
            node_to_label_dict={0: 0, 1: 1, 2: 2},
        )

        assert network.events_table is events_table
        pd.testing.assert_frame_equal(network.events_table, events_table)

    def test_csv_fast_path_preserves_order_and_index(self, tmp_path):
        """`relabel_nodes=False` does not reset CSV-loaded event tables."""
        events_table = pd.DataFrame({
            "source_nodes": [0, 1, 2],
            "target_nodes": [1, 2, 0],
            "starting_times": [2.0, 0.0, 1.0],
            "ending_times": [3.0, 1.0, 2.0],
        }, index=[20, 10, 30])
        csv_path = tmp_path / "events.csv"
        events_table.to_csv(csv_path)

        network = ContTempNetwork(
            events_table=csv_path,
            relabel_nodes=False,
            node_to_label_dict={0: 0, 1: 1, 2: 2},
            index_col=0,
        )

        assert network.events_table.index.tolist() == [20, 10, 30]
        assert network.events_table.starting_times.tolist() == [2.0, 0.0, 1.0]

    def test_inst_events_table_matches_start_plus_one_interval(self):
        """ContTempInstNetwork synthesizes ending_times = start + 1.

        The resulting events_table must equal that of an interval
        ContTempNetwork explicitly constructed with the same
        ending_times.

        Note: laplacian equality is intentionally not asserted here.
        ContTempInstNetwork.compute_laplacian_matrices implements pulse
        dynamics (state reset every step, no-op on event end), matching
        upstream TemporalNetwork.py at commit f99bca3, which is
        fundamentally distinct from the parent's interval dynamics.
        """
        starts = self.minimal.starting_times
        interval = ContTempNetwork(
            source_nodes=self.minimal.source_nodes,
            target_nodes=self.minimal.target_nodes,
            starting_times=starts,
            ending_times=[s + 1 for s in starts],
        )
        inst = ContTempInstNetwork(
            source_nodes=self.minimal.source_nodes,
            target_nodes=self.minimal.target_nodes,
            starting_times=starts,
        )
        pd.testing.assert_frame_equal(
            interval.events_table.reset_index(drop=True),
            inst.events_table.reset_index(drop=True),
        )


def test_ContTempNetworkErrors():
    with pytest.raises(AssertionError):
        ContTempNetwork(source_nodes=[0, 1], target_nodes=[1])

    with pytest.raises(Exception):
        ContTempNetwork(events_table=pd.DataFrame({"source_nodes": [0, 1]}))


def test_ContTempInstNetwork():
    """
    """
    from tempnet.temporal_network import ContTempInstNetwork
    pass


def test_compute_subspace_expm():
    """
    """
    from tempnet.temporal_network import compute_subspace_expm
    pass


def test_csc_row_normalize():
    """
    """
    from tempnet.utils import csc_row_normalize
    pass


def test_find_spectral_gap():
    """
    """
    from tempnet.utils import find_spectral_gap
    pass


def test_remove_nnz_rowcol():
    """
    """
    from tempnet.utils import remove_nnz_rowcol
    pass


def test_sparse_lapl_expm():
    """
    """
    from tempnet.temporal_network import sparse_lapl_expm
    pass


def test_set_to_ones():
    """
    """
    from tempnet.utils import set_to_ones
    pass


def test_set_to_zeroes():
    """
    """
    from tempnet.utils import set_to_zeroes
    pass


class TestRelabelNodes:
    """Tests for node relabeling done by ContTempNetwork.__init__.

    Both branches (lists and events_table) must end up with contiguous
    0..N-1 node ids in `events_table` and matching label_to_node_dict /
    node_to_label_dict.
    """

    def _make_df(self, sources, targets,
                 starts=None, ends=None):
        n = len(sources)
        if starts is None:
            starts = [float(i) for i in range(n)]
        if ends is None:
            ends = [float(i) + 1.0 for i in range(n)]
        return pd.DataFrame({
            "source_nodes": sources,
            "target_nodes": targets,
            "starting_times": starts,
            "ending_times": ends,
        })

    def test_relabel_with_events_table_non_contiguous_int_labels(self):
        df = self._make_df([10, 20, 30], [20, 30, 10])
        net = ContTempNetwork(events_table=df.copy())  # default relabel=True

        # events_table source/target columns must now be contiguous 0..N-1
        used = set(net.events_table.source_nodes.tolist()) | \
            set(net.events_table.target_nodes.tolist())
        assert used == {0, 1, 2}

        # node_to_label_dict round-trips with label_to_node_dict
        assert net.node_to_label_dict == {0: 10, 1: 20, 2: 30}
        assert net.label_to_node_dict == {10: 0, 20: 1, 30: 2}
        for n_id, lbl in net.node_to_label_dict.items():
            assert net.label_to_node_dict[lbl] == n_id

        # node_array uses the new ids
        assert net.node_array.tolist() == [0, 1, 2]
        assert net.num_nodes == 3

    def test_relabel_with_events_table_string_labels(self):
        df = self._make_df(["a", "c", "b"], ["c", "b", "a"])
        net = ContTempNetwork(events_table=df.copy())

        # All ids contiguous 0..N-1 (so subsequent matrix ops work)
        used = set(net.events_table.source_nodes.tolist()) | \
            set(net.events_table.target_nodes.tolist())
        assert used == {0, 1, 2}
        # mapping is sorted alphabetically: a->0, b->1, c->2
        assert net.node_to_label_dict == {0: "a", 1: "b", 2: "c"}

        # smoke: laplacian computation should work with relabelled ids
        net.compute_laplacian_matrices()
        assert len(net.laplacians) > 0

    def test_relabel_consistency_lists_vs_events_table(self):
        sources = [10, 20, 30]
        targets = [20, 30, 10]
        starts = [0.0, 1.0, 2.0]
        ends = [1.0, 2.0, 3.0]

        net_lists = ContTempNetwork(
            source_nodes=sources, target_nodes=targets,
            starting_times=starts, ending_times=ends,
        )
        df = self._make_df(sources, targets, starts, ends)
        net_df = ContTempNetwork(events_table=df)

        assert net_lists.node_to_label_dict == net_df.node_to_label_dict
        assert net_lists.label_to_node_dict == net_df.label_to_node_dict

        cols = ["source_nodes", "target_nodes",
                "starting_times", "ending_times"]
        pd.testing.assert_frame_equal(
            net_lists.events_table[cols].reset_index(drop=True),
            net_df.events_table[cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_relabel_off_with_events_table_preserves_ids(self):
        df = self._make_df([10, 20], [20, 10])
        original = df.copy()
        provided = {0: "x", 1: "y"}  # arbitrary user-supplied dict
        net = ContTempNetwork(
            events_table=df,
            relabel_nodes=False,
            node_to_label_dict=provided,
        )
        # events_table columns are unchanged
        pd.testing.assert_series_equal(
            net.events_table.source_nodes, original.source_nodes,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            net.events_table.target_nodes, original.target_nodes,
            check_names=False,
        )
        # provided node_to_label_dict preserved; no label_to_node_dict built
        assert net.node_to_label_dict is provided
        assert not hasattr(net, "label_to_node_dict")

    def test_events_table_from_csv_path_relabels(self, tmp_path):
        df = self._make_df([10, 20, 30], [20, 30, 10])
        csv_path = tmp_path / "events.csv"
        df.to_csv(csv_path, index=False)

        net = ContTempNetwork(events_table=csv_path)
        used = set(net.events_table.source_nodes.tolist()) | \
            set(net.events_table.target_nodes.tolist())
        assert used == {0, 1, 2}
        assert net.node_to_label_dict == {0: 10, 1: 20, 2: 30}

    def test_relabel_does_not_mutate_caller_dataframe(self):
        df = self._make_df([10, 20, 30], [20, 30, 10])
        df_before = df.copy()
        ContTempNetwork(events_table=df)  # default relabel=True
        # caller's df should be unchanged
        pd.testing.assert_frame_equal(df, df_before)


class TestContTempInstNetwork:
    """Tests for the ContTempInstNetwork constructor.

    Instant networks are described by (source, target, starting_time)
    triplets only. The constructor must be able to accept events_tables
    that lack the ending_times column (DataFrame or CSV path) and must
    synthesize one before delegating to the parent constructor. All
    durations should be set to 1.0 and instantaneous_events flag set.
    """

    def _make_inst_df(self, sources, targets, starts=None):
        n = len(sources)
        if starts is None:
            starts = [float(i) for i in range(n)]
        return pd.DataFrame({
            "source_nodes": sources,
            "target_nodes": targets,
            "starting_times": starts,
        })

    def test_init_from_dataframe_synthesizes_ending_times(self):
        df = self._make_inst_df([0, 1, 2], [1, 2, 0],
                                starts=[0.0, 1.0, 2.0])
        net = ContTempInstNetwork(events_table=df)

        assert "ending_times" in net.events_table.columns
        # ending_times derived from sorted unique starts; last one += 1
        assert net.events_table.ending_times.tolist() == [1.0, 2.0, 3.0]
        assert net.events_table["durations"].tolist() == [1.0, 1.0, 1.0]
        assert net.instantaneous_events is True

    def test_init_from_csv_path_synthesizes_ending_times(self, tmp_path):
        df = self._make_inst_df([10, 20, 30], [20, 30, 10],
                                starts=[0.0, 1.0, 2.0])
        csv_path = tmp_path / "inst_events.csv"
        df.to_csv(csv_path, index=False)

        net = ContTempInstNetwork(events_table=csv_path)

        assert "ending_times" in net.events_table.columns
        assert net.events_table.ending_times.tolist() == [1.0, 2.0, 3.0]
        # default relabel_nodes=True remaps 10/20/30 to 0/1/2
        used = set(net.events_table.source_nodes.tolist()) | \
            set(net.events_table.target_nodes.tolist())
        assert used == {0, 1, 2}
        assert net.node_to_label_dict == {0: 10, 1: 20, 2: 30}

    def test_init_from_positional_args(self):
        net = ContTempInstNetwork(
            source_nodes=[0, 1, 2],
            target_nodes=[1, 2, 0],
            starting_times=[0.0, 1.0, 2.0],
        )
        assert net.instantaneous_events is True
        assert net.events_table["durations"].tolist() == [1.0, 1.0, 1.0]
        # Should be runnable end-to-end
        net.compute_laplacian_matrices()
        assert len(net.laplacians) > 0

    def test_init_with_existing_ending_times_column_preserved(self):
        df = self._make_inst_df([0, 1, 2], [1, 2, 0],
                                starts=[0.0, 1.0, 2.0])
        df["ending_times"] = [10.0, 20.0, 30.0]
        net = ContTempInstNetwork(events_table=df)
        # Existing ending_times must not be overwritten
        assert net.events_table.ending_times.tolist() == [10.0, 20.0, 30.0]

    def test_does_not_mutate_caller_dataframe(self):
        df = self._make_inst_df([0, 1, 2], [1, 2, 0],
                                starts=[0.0, 1.0, 2.0])
        df_before = df.copy()
        ContTempInstNetwork(events_table=df)
        pd.testing.assert_frame_equal(df, df_before)

    def test_uneven_starts_use_start_plus_one(self):
        """Synthesized ending_times must be start + 1 for each event,
        independent of the spacing or uniqueness of starting times.
        """
        net = ContTempInstNetwork(
            source_nodes=[0, 1, 2],
            target_nodes=[1, 2, 0],
            starting_times=[0.0, 0.5, 5.0],
        )
        assert net.events_table.ending_times.tolist() == [1.0, 1.5, 6.0]
        assert net.events_table["durations"].tolist() == [1.0, 1.0, 1.0]

    def test_uneven_starts_via_dataframe_use_start_plus_one(self):
        df = self._make_inst_df([0, 1, 2], [1, 2, 0],
                                starts=[0.0, 0.5, 5.0])
        net = ContTempInstNetwork(events_table=df)
        assert net.events_table.ending_times.tolist() == [1.0, 1.5, 6.0]

    def test_unsorted_dataframe_default_path_sorts_and_resets_index(self):
        df = self._make_inst_df([0, 1, 2], [1, 2, 0],
                                starts=[2.0, 0.0, 1.0])
        df.index = [20, 10, 30]

        net = ContTempInstNetwork(events_table=df)

        assert net.events_table.starting_times.tolist() == [0.0, 1.0, 2.0]
        assert net.events_table.ending_times.tolist() == [1.0, 2.0, 3.0]
        assert net.events_table.index.tolist() == [0, 1, 2]

    def test_dataframe_fast_path_preserves_order_index_and_identity(self):
        df = pd.DataFrame({
            "source_nodes": [0, 1, 2],
            "target_nodes": [1, 2, 0],
            "starting_times": [0.0, 1.0, 2.0],
            "ending_times": [1.0, 2.0, 3.0],
        })

        net = ContTempInstNetwork(
            events_table=df,
            relabel_nodes=False,
            node_to_label_dict={0: 0, 1: 1, 2: 2},
        )

        assert net.events_table is df
        pd.testing.assert_frame_equal(net.events_table, df)


class TestContTempNetworkEndingTimesRequired:
    """ContTempNetwork must raise ValueError when ending_times is absent.

    The validation pinpoints the missing input ('ending_times' or the
    DataFrame column) and directs users to ContTempInstNetwork for
    instantaneous networks.
    """

    def test_positional_none_ending_raises(self):
        with pytest.raises(ValueError) as exc:
            ContTempNetwork(
                source_nodes=[0, 1], target_nodes=[1, 0],
                starting_times=[0.0, 1.0], ending_times=None,
            )
        msg = str(exc.value)
        assert "ending_times" in msg
        assert "ContTempInstNetwork" in msg

    def test_positional_empty_ending_raises(self):
        with pytest.raises(ValueError) as exc:
            ContTempNetwork(
                source_nodes=[0, 1], target_nodes=[1, 0],
                starting_times=[0.0, 1.0], ending_times=[],
            )
        assert "ending_times" in str(exc.value)

    def test_events_table_missing_ending_column_raises(self):
        df = pd.DataFrame({
            "source_nodes": [0, 1],
            "target_nodes": [1, 0],
            "starting_times": [0.0, 1.0],
        })
        with pytest.raises(ValueError) as exc:
            ContTempNetwork(events_table=df)
        msg = str(exc.value)
        assert "ending_times" in msg
        assert "ContTempInstNetwork" in msg

    def test_all_empty_inputs_build_empty_network(self):
        """Degenerate all-empty case is valid (0 events), not an error:
        the ending_times validator only fires when starts are present.
        """
        net = ContTempNetwork(
            source_nodes=[], target_nodes=[],
            starting_times=[], ending_times=[],
        )
        assert net.num_events == 0

# --------------------------------------------------------------------------- #
# Node relabeling and merge events
# --------------------------------------------------------------------------- #
class TestRelabelNodesAndMergeEvents:
    """(A,B,0,3) and (A,B,1,4) should collapse into a single (A,B,0,4) event,
    for both relabel and no-relabel construction (parametrized fixture).
     
    Both branches (lists and events_table) must end up with contiguous
    0..N-1 node ids in events_table and matching label/node dicts.
    """
    def test_relabel_events_table_non_contiguous_int_labels(self):
        
        df = make_df(sources=[10, 20, 30], targets=[20, 30, 10], starts=[0, 1, 2], ends=[1, 2, 3])
        net = ContTempNetwork(events_table=df.copy())  # default relabel=True

        used = set(net.events_table.source_nodes) | set(net.events_table.target_nodes)
        assert used == {0, 1, 2}

        assert net.node_to_label_dict == {0: 10, 1: 20, 2: 30}
        assert net.label_to_node_dict == {10: 0, 20: 1, 30: 2}
        for n_id, lbl in net.node_to_label_dict.items():
            assert net.label_to_node_dict[lbl] == n_id

        assert net.node_array.tolist() == [0, 1, 2]
        assert net.num_nodes == 3

    def test_relabel_events_table_string_labels(self):
        df = make_df(sources=["a", "c", "b"], targets=["c", "b", "a"], starts=[0, 1, 2], ends=[1, 2, 3])
        net = ContTempNetwork(events_table=df.copy())

        used = set(net.events_table.source_nodes) | set(net.events_table.target_nodes)
        assert used == {0, 1, 2}
        # mapping is sorted alphabetically: a->0, b->1, c->2
        assert net.node_to_label_dict == {0: "a", 1: "b", 2: "c"}

        # laplacian computation should work with relabelled ids
        net.compute_laplacian_matrices()
        assert len(net.laplacians) > 0

    def test_relabel_consistency_lists_vs_events_table(self):
        sources, targets = [10, 20, 30], [20, 30, 10]
        starts, ends = [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]

        net_lists = ContTempNetwork(
            source_nodes=sources, target_nodes=targets,
            starting_times=starts, ending_times=ends,
        )
        net_df = ContTempNetwork(events_table=make_df(sources=sources, targets=targets, starts=starts, ends=ends))

        assert net_lists.node_to_label_dict == net_df.node_to_label_dict
        assert net_lists.label_to_node_dict == net_df.label_to_node_dict

        cols = ["source_nodes", "target_nodes", "starting_times", "ending_times"]
        pd.testing.assert_frame_equal(
            net_lists.events_table[cols].reset_index(drop=True),
            net_df.events_table[cols].reset_index(drop=True),
            check_dtype=False,
        )

    def test_relabel_off_preserves_provided_ids(self):
        df = make_df(["x", "y"], ["y", "x"], starts=[10, 25], ends=[20, 30])
        provided = {"x": 0, "y": 1}
        net = ContTempNetwork(events_table=df, label_to_node_dict=provided)

        np.testing.assert_array_equal(net.events_table.source_nodes.values, [0, 1])
        np.testing.assert_array_equal(net.events_table.target_nodes.values, [1, 0])
        assert net.label_to_node_dict is provided
        assert hasattr(net, "node_to_label_dict")

    def test_relabel_off_non_unique_labels_raises(self):
        df = make_df(["x", "y"], ["y", "x"], starts=[10, 25], ends=[20, 30])
        with pytest.raises(ValueError):
            ContTempNetwork(events_table=df, label_to_node_dict={"x": 0, "y": 0})

    def test_relabel_off_non_canonical_labels_raises(self):
        df = make_df(["x", "y"], ["y", "x"], starts=[10, 25], ends=[20, 30])
        with pytest.raises(ValueError):
            ContTempNetwork(events_table=df, label_to_node_dict={"x": 0, "y": 2})

    def test_relabel_does_not_mutate_caller_dataframe(self):
        df = make_df(sources=[10, 20, 30], targets=[20, 30, 10], starts=[0, 1, 2], ends=[1, 2, 3])
        df_before = df.copy()
        ContTempNetwork(events_table=df)  # default relabel=True
        pd.testing.assert_frame_equal(df, df_before)

    def test_events_table_from_csv_path_relabels(self, tmp_path):
        csv_path = tmp_path / "events.csv"
        make_df(sources=[10, 20, 30], targets=[20, 30, 10], starts=[0, 1, 2], ends=[1, 2, 3]).to_csv(csv_path, index=False)
        net = ContTempNetwork(events_table=csv_path)

        used = set(net.events_table.source_nodes) | set(net.events_table.target_nodes)
        assert used == {0, 1, 2}
        assert net.node_to_label_dict == {0: 10, 1: 20, 2: 30}

    @pytest.mark.parametrize("input_mode", ["lists", "events_table"])
    @pytest.mark.parametrize(
        "sources, targets",
        [
            ([1, 2, 3], [2, 3, 1]),
            ([0, 2, 3], [2, 3, 0]),
            ([10, 20, 30], [20, 30, 10]),
            (["a", "b", "c"], ["b", "c", "a"]),
        ],
    )
    def test_bad_node_labels_are_safe_after_relabel(
        self,
        input_mode,
        sources,
        targets,
    ):
        starts = [0.0, 1.0, 2.0]
        ends = [1.0, 2.0, 3.0]

        if input_mode == "lists":
            net = ContTempNetwork(
                source_nodes=sources,
                target_nodes=targets,
                starting_times=starts,
                ending_times=ends,
            )
        else:
            net = ContTempNetwork(
                events_table=make_df(
                    sources=sources,
                    targets=targets,
                    starts=starts,
                    ends=ends,
                )
            )

        used = set(net.events_table.source_nodes) | set(
            net.events_table.target_nodes
        )
        expected_nodes = set(range(net.num_nodes))

        assert used == expected_nodes
        assert net.node_array.tolist() == list(range(net.num_nodes))

        A = net.compute_static_adjacency_matrix()
        assert A.shape == (net.num_nodes, net.num_nodes)

        net.compute_laplacian_matrices()
        assert all(
            L.shape == (net.num_nodes, net.num_nodes)
            for L in net.laplacians
        )

        net.compute_inter_transition_matrices(
            lamda=1.0,
            method="dense_expm",
        )
        assert all(
            T.shape == (net.num_nodes, net.num_nodes)
            for T in net.inter_T[1.0]
        )

        net.compute_transition_matrices(lamda=1.0, force_csr=True)
        assert net.T[1.0][-1].shape == (net.num_nodes, net.num_nodes)

    def test_overlapping_events_are_merged(self, network_overlapping):
        assert network_overlapping.num_events == 2

    def test_merged_event_span(self, network_overlapping):
        row = network_overlapping.events_table.iloc[0]
        assert row["starting_times"] == 0
        assert row["ending_times"] == 4

    def test_merge_flag_set(self, network_overlapping):
        assert network_overlapping._overlapping_events_merged is True
        
# --------------------------------------------------------------------------- #
# Instantaneous networks
# --------------------------------------------------------------------------- #
class TestContTempInstNetwork:
    """ContTempInstNetwork accepts (source, target, starting_time) triplets
    only, and synthesizes ending_times = starting_times before delegating to
    the parent constructor.
    """

    def test_constructor_wrong_columns(self):
        df = pd.DataFrame({
            "source_nodes": [0, 1],
            "target_nodes": [1, 0],
        })
        with pytest.raises(ValueError):
            ContTempInstNetwork(events_table=df)


    def test_constructor_wrong_file_type(self, tmp_path):
        with pytest.raises(ValueError):
            ContTempInstNetwork(events_table=1)
 
    def test_init_from_dataframe_synthesizes_ending_times(self):
        df = make_df(sources=[0, 1, 2], targets=[1, 2, 0], starts=[0.0, 1.0, 2.0])
        net = ContTempInstNetwork(events_table=df)
        assert "ending_times" in net.events_table.columns
        assert net.events_table.ending_times.tolist() == [0.0, 1.0, 2.0]

    def test_init_from_csv_path_synthesizes_ending_times(self, tmp_path):
        csv_path = tmp_path / "inst_events.csv"
        make_df(sources=[10, 20, 30], targets=[20, 30, 10], starts=[0.0, 1.0, 2.0]).to_csv(
            csv_path, index=False,
        )
        net = ContTempInstNetwork(events_table=csv_path)

        assert "ending_times" in net.events_table.columns
        assert net.events_table.ending_times.tolist() == [0.0, 1.0, 2.0]
        used = set(net.events_table.source_nodes) | set(net.events_table.target_nodes)
        assert used == {0, 1, 2}
        assert net.node_to_label_dict == {0: 10, 1: 20, 2: 30}

    def test_init_from_positional_args(self):
        net = ContTempInstNetwork(
            source_nodes=[0, 1, 2], target_nodes=[1, 2, 0],
            starting_times=[0.0, 1.0, 2.0],
        )
        net.compute_laplacian_matrices()
        assert len(net.laplacians) > 0

    def test_existing_ending_times_column_preserved(self):
        df = make_df(sources=[0, 1, 2], targets=[1, 2, 0], starts=[0.0, 1.0, 2.0])
        df["ending_times"] = [10.0, 20.0, 30.0]
        net = ContTempInstNetwork(events_table=df)
        assert net.events_table.ending_times.tolist() == [0.0, 1.0, 2.0]

    def test_does_not_mutate_caller_dataframe(self):
        df = make_df(sources=[0, 1, 2], targets=[1, 2, 0], starts=[0.0, 1.0, 2.0])
        df_before = df.copy()
        ContTempInstNetwork(events_table=df)
        pd.testing.assert_frame_equal(df, df_before)

    def test_uneven_starts_use_start_positional(self):
        net = ContTempInstNetwork(
            source_nodes=[0, 1, 2], target_nodes=[1, 2, 0], starting_times=[0.0, 0.5, 5.0],
        )
        assert net.events_table.ending_times.tolist() == [0.0, 0.5, 5.0]

    def test_uneven_starts_use_start_dataframe(self):
        net = ContTempInstNetwork(
            events_table=make_df(sources=[0, 1, 2], targets=[1, 2, 0], starts=[0.0, 0.5, 5.0]),
        )
        assert net.events_table.ending_times.tolist() == [0.0, 0.5, 5.0]
    
    EXPM_METHODS = ["sparse_expm", "parallel_expm", "mfp_exp"]
    @pytest.mark.parametrize("method", EXPM_METHODS)
    @pytest.mark.parametrize("lamda", [0.1, 1.0, 5.0])
    def test_method_matches_dense_exact(self, method, lamda):
        """Every method should match dense_expm to near machine precision."""
        net = ContTempInstNetwork(
            events_table=make_df(sources=[0, 1, 2], targets=[1, 2, 0], starts=[0.0, 0.5, 0.5]),
        )
        net.compute_laplacian_matrices()
        for k, L in enumerate(net.laplacians):
            tau =1
            T_ref = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, "dense_expm"))
            T = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, method))
            np.testing.assert_allclose(
                T, T_ref, rtol=1e-6, atol=1e-6,
                err_msg=f"{method} != dense_expm at step {k}, lamda={lamda}",
            )

# --------------------------------------------------------------------------- #
# Test save/load 
# --------------------------------------------------------------------------- #
class TestSaveLoad:

    def test_save_load(self, simple_network,tmp_path):
        simple_network.compute_laplacian_matrices()
        simple_network.compute_inter_transition_matrices(lamda=1)
        simple_network.compute_transition_matrices(
                                    lamda=1,
                                    save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    tol=None)
        simple_network.save(tmp_path / "simple_network.pkl")
        

        new_network=ContTempNetwork.load(tmp_path / "simple_network.pkl")

        for col in ["source_nodes", "target_nodes", "starting_times", "ending_times"]:
            pd.testing.assert_series_equal(
                simple_network.events_table[col], new_network.events_table[col],
            )

    EXPM_METHODS = ["sparse_expm", "parallel_expm", "mfp_exp"]
    @pytest.mark.parametrize("method", EXPM_METHODS)
    def test_save_load_T(self, simple_network,tmp_path, method):
        simple_network.compute_laplacian_matrices()
        for lam in [1, 10, 0.1]:
            simple_network.compute_inter_transition_matrices(lamda=lam, method=method)
            simple_network.compute_transition_matrices(
                                        lamda=lam,
                                        save_intermediate=False,
                                        reverse_time=False,
                                        force_csr=True,
                                        tol=None)
        simple_network.save_T(tmp_path / "simple_network.pickle")
        

        T=ContTempNetwork.load_T(tmp_path / "simple_network.pickle")

        assert type(T)==dict # it might be really weak...

class TestBasicProperties:

    def test_num_nodes(self, simple_network):
        assert simple_network.num_nodes == 3

    def test_num_events(self, simple_network):
        assert simple_network.num_events == 4

    def test_start_time(self, simple_network):
        assert simple_network.start_time == 0

    def test_end_time(self, simple_network):
        assert simple_network.end_time == 7

    def test_print(self ,simple_network):
        s=str(simple_network)
        assert s==  "<class 'tempnet.temporal_network.ContTempNetwork'> with 3 nodes and 4 events" 

    def test_node_array_sorted(self, simple_network):
        assert list(simple_network.node_array) == [0, 1, 2]
        assert simple_network.nodes==['A', 'B', 'C']

    def test_durations_column_exists(self, simple_network):
        assert "durations" in simple_network.events_table.columns

    def test_durations_values(self, simple_network):
        assert list(simple_network.events_table["durations"]) == [2, 2, 1, 1]

    def test_events_sorted_by_start(self, simple_network):
        starts = simple_network.events_table["starting_times"].tolist()
        assert starts == sorted(starts)

    def test_required_columns_present(self, simple_network):
        for col in ["source_nodes", "target_nodes", "starting_times",
                    "ending_times", "durations"]:
            assert col in simple_network.events_table.columns

    def test_active_edges(self, simple_network):
        assert simple_network.num_active_edges(t_start=None, t_end=None)==4
        assert simple_network.num_active_edges(t_start=1, t_end=2)==2
        assert simple_network.num_active_edges(t_start=None, t_end=5)==3
        assert simple_network.num_active_edges(t_start=6.5, t_end=None)==1
        assert simple_network.num_active_edges(t_start=3.5, t_end=3.75)==0


    def test_active_nodes(self, simple_network):
        assert simple_network.num_active_nodes(t_start=None, t_end=None)==3
        assert simple_network.num_active_nodes(t_start=1, t_end=2)==3
        assert simple_network.num_active_nodes(t_start=None, t_end=5)==3
        assert simple_network.num_active_nodes(t_start=6.5, t_end=None)==2
        assert simple_network.num_active_nodes(t_start=3.5, t_end=3.75)==0



    def test_index_reset(self, simple_network):
        assert list(simple_network.events_table.index) == list(
            range(simple_network.num_events)
        )
        
    def test_adj_full(self, simple_network):
        A = simple_network.compute_static_adjacency_matrix().toarray()
        expected = np.array([
            [0, 3, 1],
            [3, 0, 2],
            [1, 2, 0],
        ])
        assert np.allclose(A, expected)
    def test_adj_window_0_2(self, simple_network):
        A = simple_network.compute_static_adjacency_matrix(
            start_time=0, end_time=2,
        ).toarray()
        expected = np.array([
            [0, 2, 0],
            [2, 0, 1],
            [0, 1, 0],
        ])
        assert np.allclose(A, expected)

    def test_adj_window_2p5_3(self, simple_network):
        A = simple_network.compute_static_adjacency_matrix(
            start_time=2.5, end_time=3,
        ).toarray()
        expected = np.array([
            [0, 0, 0],
            [0, 0, 0.5],
            [0, 0.5, 0],
        ])
        assert np.allclose(A, expected)

    @pytest.mark.parametrize("dynamics", [ "rw", "heat"])
    def test_laplacians_count(self, simple_network, dynamics):
        """One Laplacian per inter-event step over the full grid.

        times = [0,1,2,3,4,5,6,7] -> 7 inter-event steps.
        """
        simple_network.compute_laplacian_matrices(dynamics=dynamics)
        assert len(simple_network.laplacians) == 7

    def test_laplacian_step_1_2_randomwalk(self, simple_network):
        """Step [1,2]: A-B and B-C active. Random-walk Laplacian I - D^-1 A."""
        simple_network.compute_laplacian_matrices(dynamics="rw")
        L = simple_network.laplacians[1].toarray()
        expected = np.array([
            [1.0, -1.0, 0.0],
            [-0.5, 1.0, -0.5],
            [0.0, -1.0, 1.0],
        ])
        assert np.allclose(L, expected)


    def test_laplacian_step_1_2_heat(self, simple_network):
        """Step [1,2]: A-B and B-C active. Heat kernel Laplacian."""
        simple_network.compute_laplacian_matrices(dynamics="heat")
        L = simple_network.laplacians[1].toarray()
        expected = np.array([
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ])
        assert np.allclose(L, expected)


    def test_laplacian_step_0_1_connected_block(self, simple_network):
        """Step [0,1]: only A-B active; C isolated. C's diagonal is
        convention-dependent, so only the A-B block and zero coupling to C
        are asserted.
        """
        simple_network.compute_laplacian_matrices()
        L = simple_network.laplacians[0].toarray()
        assert L[0, 0] == 1
        assert L[0, 1] == -1
        assert L[1, 0] == -1
        assert L[1, 1] == 1
        assert L[0, 2] == 0
        assert L[1, 2] == 0
        assert L[2, 0] == 0
        assert L[2, 1] == 0
        assert L[2, 2] == 0  # self loop

    @pytest.mark.parametrize("dynamics", [ "rw", "heat"])
    def test_laplacian_empty_step_all_zero(self, simple_network, dynamics):
        """Step [3,4]: no active events, so the Laplacian is all zeros."""
        simple_network.compute_laplacian_matrices(dynamics=dynamics)
        L = simple_network.laplacians[3].toarray()
        assert np.all(L == 0)

    @pytest.mark.parametrize("dynamics", [ "rw", "heat"])
    def test_laplacian_rows_sum_zero_connected_step(self, simple_network, dynamics):
        simple_network.compute_laplacian_matrices(dynamics=dynamics)
        n = simple_network.num_nodes
        for i in range(len(simple_network.laplacians)):
            L = simple_network.laplacians[i].toarray()
            assert L.shape == (n, n)
            assert np.allclose(L.sum(axis=1), 0.0)

# --------------------------------------------------------------------------- #
# Transition matrices
# --------------------------------------------------------------------------- #
class TestTransitionMatrices:

    EXPM_METHODS = ["sparse_expm", "parallel_expm", "mfp_exp"]

    @pytest.mark.parametrize("method", EXPM_METHODS)
    @pytest.mark.parametrize("lamda", [0.1, 1.0, 5.0])
    def test_method_matches_dense_exact(self, simple_network, method, lamda):
        """Every method should match dense_expm to near machine precision."""
        net = simple_network
        net.compute_laplacian_matrices()
        for k, L in enumerate(net.laplacians):
            tau = net.times[k + 1] - net.times[k]
            T_ref = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, "dense_expm"))
            T = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, method))
            np.testing.assert_allclose(
                T, T_ref, rtol=1e-6, atol=1e-6,
                err_msg=f"{method} != dense_expm at step {k}, lamda={lamda}",
            )


    def test_no_laplacian(self, simple_network):
        net = simple_network
        lamda=1
        with pytest.raises(RuntimeError):
            net.compute_inter_transition_matrices(lamda=lamda)

    def test_wrong_method(self, simple_network):
        net = simple_network
        lamda=1
        net.compute_laplacian_matrices()
        with pytest.raises(Exception):
            for k, L in enumerate(net.laplacians):
                tau = net.times[k + 1] - net.times[k]
                net._compute_single_T(L, tau, lamda, net.num_nodes, "method")

    @pytest.mark.parametrize("method", EXPM_METHODS)
    def test_transition_matrix_is_stochastic(self, simple_network, method):
        """expm of a (negative) Laplacian gives row-stochastic matrices."""
        net = simple_network
        lamda = 1.0
        net.compute_laplacian_matrices()
        for k, L in enumerate(net.laplacians):
            tau = net.times[k + 1] - net.times[k]
            T = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, method))
            np.testing.assert_allclose(T.sum(axis=1), np.ones(net.num_nodes), atol=1e-10)
            assert (T >= 0).all(), f"{method} produced negative entries at step {k}"

    def test_mfp_exp_error_decreases_with_tighter_tolerance(self, simple_network):
        """Tighter err in mfp_exp should not increase the gap to dense_expm."""
        net = simple_network
        net.compute_laplacian_matrices()
        lamda = 1.0
        L = net.laplacians[0]
        tau = net.times[1] - net.times[0]

        T_ref = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, "dense_expm"))
        T_loose = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, "mfp_exp", err=1e-4))
        T_tight = to_dense(net._compute_single_T(L, tau, lamda, net.num_nodes, "mfp_exp", err=1e-10))

        mae_loose = np.mean(np.abs(T_loose - T_ref))
        mae_tight = np.mean(np.abs(T_tight - T_ref))
        assert mae_tight <= mae_loose + 1e-12


# --------------------------------------------------------------------------- #
# Windowed (t_start/t_stop) transition matrices
# --------------------------------------------------------------------------- #
class TestWindowedTransitionMatrices:
    """Inter-event transition matrices must use taus from the Laplacian window.

    ``compute_laplacian_matrices(t_start=..., t_stop=...)`` restricts the
    computation to a sub-range of the event-time grid: ``self.laplacians[j]``
    then corresponds to the time step
    ``[times[k0 + j], times[k0 + j + 1]]`` with
    ``k0 = self._k_start_laplacians``.

    ``compute_inter_transition_matrices`` must therefore pair each Laplacian
    with the inter-event time of *its own* step,
    ``tau_j = times[k0 + j + 1] - times[k0 + j]``.

    The current implementation instead computes
    ``taus[j] = times[j + 1] - times[j]`` (i.e. always starting from
    ``times[0]``), so as soon as ``k0 > 0`` every transition matrix
    ``T_j = expm(-tau_j * lamda * L_j)`` is built with the tau of the wrong
    time step. These tests capture that erroneous behaviour (they are *red*
    until the tau indexing is fixed):

    1. ``test_windowed_inter_T_uses_window_taus`` pins the general contract:
       the inter_T sequence of a windowed computation must equal the
       corresponding slice of the inter_T sequence computed over the full
       time range.
    2. ``test_windowed_tau_matches_expm_directly`` checks the first windowed
       transition matrix against an explicitly computed
       ``expm(-tau0 * lamda * L_0)`` with ``tau0`` taken at the window start.

    The event times are chosen so that the inter-event gaps are non-uniform
    (grid ``[0, 2, 10, 11, 25, 26]``); with uniform gaps the misaligned taus
    would accidentally produce correct results.
    """

    LAMDA = 1.0

    @pytest.fixture
    def uneven_events(self):
        """Events with non-uniform gaps: times grid [0, 2, 10, 11, 25, 26]."""
        return make_df(
            sources=[0, 1, 0],
            targets=[1, 2, 2],
            starts=[0, 10, 25],
            ends=[2, 11, 26],
        )

    def test_windowed_inter_T_uses_window_taus(self, uneven_events):
        full = ContTempNetwork(events_table=uneven_events)
        full.compute_laplacian_matrices()
        full.compute_inter_transition_matrices(lamda=self.LAMDA)

        win = ContTempNetwork(events_table=uneven_events)
        win.compute_laplacian_matrices(t_start=10, t_stop=26)
        win.compute_inter_transition_matrices(lamda=self.LAMDA)

        k0 = win._k_start_laplacians  # offset into the full time grid
        assert k0 > 0, "fixture must exercise a window with k_start > 0"
        assert len(win.inter_T[self.LAMDA]) == len(win.laplacians)

        for j, T_win in enumerate(win.inter_T[self.LAMDA]):
            T_full = full.inter_T[self.LAMDA][k0 + j]
            np.testing.assert_allclose(
                to_dense(T_win), to_dense(T_full), atol=1e-12,
                err_msg=f"window step {j} != full step {k0 + j}",
            )

    def test_windowed_tau_matches_expm_directly(self, uneven_events):
        from scipy.linalg import expm as dense_expm

        net = ContTempNetwork(events_table=uneven_events)
        net.compute_laplacian_matrices(t_start=10, t_stop=26)
        net.compute_inter_transition_matrices(lamda=self.LAMDA)

        k0 = net._k_start_laplacians
        tau0 = net.times[k0 + 1] - net.times[k0]
        expected = dense_expm(
            -tau0 * self.LAMDA * net.laplacians[0].toarray()
        )
        np.testing.assert_allclose(
            to_dense(net.inter_T[self.LAMDA][0]), expected, atol=1e-12,
        )


# --------------------------------------------------------------------------- #
# inter_T must not be mutated by derived computations
# --------------------------------------------------------------------------- #
class TestInterTNotMutated:
    """``compute_transition_matrices`` must not modify ``self.inter_T``.

    ``self.inter_T[lamda]`` holds the inter-event transition matrices, the
    primary data from which the accumulated transition matrices ``self.T``
    are *derived*. Computing a derived quantity must therefore leave
    ``inter_T`` untouched.

    The current implementation's ``clean()`` helper (see
    ``compute_transition_matrices``) calls ``set_to_zeroes(Tk, tol)`` and
    ``inplace_csr_row_normalize(Tk)`` directly on the matrices stored in
    ``self.inter_T[lamda]``. Even the seed matrix is affected, because
    ``inter[k_init].tocsr()`` returns the stored object itself when it is
    already CSR (SciPy only copies with ``copy=True``). Consequences:

    * ``inter_T`` silently loses precision: entries below the caller's
      ``tol`` are zeroed *in the stored input data*,
    * anything using ``inter_T`` afterwards (``save_inter_T``, plotting,
      recomputation) operates on altered matrices,
    * results become dependent on how many times, and with which ``tol``,
      ``compute_transition_matrices`` was previously called on the same
      instance (non-idempotence).

    These tests are *red* until ``clean()`` operates on copies, or cleaning
    is applied only to the accumulated product and never to the stored
    inter-event factors.
    """

    LAMDA = 1.0
    COARSE_TOL = 1e-2   # coarse on purpose: makes the in-place zeroing visible

    @pytest.fixture
    def net_with_inter_T(self):
        """Network with one very short inter-event step (tau = 1e-4).

        The step [1, 1.0001] yields a transition matrix close to the
        identity, with off-diagonal entries of order 1e-4. Those entries
        fall below the relative threshold ``COARSE_TOL * max|T|`` used by
        ``set_to_zeroes``, so an in-place ``clean()`` visibly zeroes stored
        ``inter_T`` data (beyond mere normalization round-off).
        """
        net = ContTempNetwork(
            source_nodes=["A", "B", "A", "A"],
            target_nodes=["B", "C", "C", "B"],
            starting_times=[0, 1, 1.0001, 6],
            ending_times=[2, 3, 5, 7],
        )
        net.compute_laplacian_matrices()
        net.compute_inter_transition_matrices(lamda=self.LAMDA)
        return net

    def test_inter_T_unchanged_by_compute_transition_matrices(
        self, net_with_inter_T,
    ):
        """Byte-for-byte: inter_T must be identical before and after."""
        net = net_with_inter_T
        before = [to_dense(T).copy() for T in net.inter_T[self.LAMDA]]

        net.compute_transition_matrices(
            lamda=self.LAMDA, force_csr=True, tol=self.COARSE_TOL,
        )

        for k, (b, T) in enumerate(zip(before, net.inter_T[self.LAMDA])):
            np.testing.assert_array_equal(
                to_dense(T), b,
                err_msg=(
                    f"inter_T[{k}] was mutated by"
                    " compute_transition_matrices"
                ),
            )

    def test_transition_matrices_idempotent_wrt_previous_runs(
        self, net_with_inter_T,
    ):
        """T computed after a coarse-tol run must equal a fresh computation.

        With the in-place mutation, the first (coarse-tol) run zeroes small
        entries of ``inter_T``, so the second run with ``tol=None`` on the
        same instance no longer reproduces the result obtained on a fresh
        network.
        """
        net = net_with_inter_T
        net.compute_transition_matrices(
            lamda=self.LAMDA, force_csr=True, tol=self.COARSE_TOL,
        )
        del net.T          # force full recomputation on the same instance
        net.compute_transition_matrices(
            lamda=self.LAMDA, force_csr=True, tol=None,
        )

        fresh = ContTempNetwork(events_table=net.events_table)
        fresh.compute_laplacian_matrices()
        fresh.compute_inter_transition_matrices(lamda=self.LAMDA)
        fresh.compute_transition_matrices(
            lamda=self.LAMDA, force_csr=True, tol=None,
        )

        np.testing.assert_allclose(
            to_dense(net.T[self.LAMDA][-1]),
            to_dense(fresh.T[self.LAMDA][-1]),
            atol=1e-14,
            err_msg=(
                "tol=None run polluted by earlier coarse-tol mutation"
                " of inter_T"
            ),
        )


# --------------------------------------------------------------------------- #
# Instantaneous (pulse) network semantics: ending_times == starting_times
# --------------------------------------------------------------------------- #
class TestInstNetworkPulseSemantics:
    """Contract of ``ContTempInstNetwork`` under zero-duration pulses.

    Adopted convention: instantaneous events are represented with
    ``ending_times == starting_times`` (zero duration). This is the honest
    representation for pulse dynamics -- the time grid consists exactly of
    the unique pulse times, and inter-event taus are the true gaps between
    pulses -- but several parts of the machinery still assume strictly
    positive durations. These tests capture the resulting defects (red
    until fixed):

    R1  Every pulse must produce a Laplacian / inter_T step -- including
        the *last* one. The Laplacian loop only iterates over grid times
        strictly smaller than ``t_stop = times[-1]``; with zero durations
        the last grid point *is* the last pulse, so its Laplacian is
        silently dropped.

    R2  Aggregating an instantaneous network into a static adjacency
        matrix must count events per node pair. The parent implementation
        aggregates event *durations*, which are all zero here, so it
        silently returns an all-zero matrix.

    R3  Activity window queries (``active_nodes``, ``num_active_nodes``,
        ``num_active_edges``) must include a pulse lying exactly at
        ``t_start``. The overlap mask ``ending_times > t_start`` excludes
        zero-duration events at the window boundary -- in particular, the
        default full window (``t_start = start_time``) misses the very
        first pulse of the network.

    R4  An ``events_table`` carrying an ``ending_times`` column that
        conflicts with the zero-duration convention must be rejected with
        a ``ValueError`` -- not silently overwritten. Silently replacing
        user data hides the mistake of loading an interval network into
        ``ContTempInstNetwork``. (Fixing this requires updating
        ``test_existing_ending_times_column_preserved``, which currently
        pins the silent overwrite.)
    """

    @pytest.fixture
    def pulse_network(self):
        """Three pulses at t = 0, 1, 5: (A,B), (B,C), (A,B)."""
        return ContTempInstNetwork(
            source_nodes=["A", "B", "A"],
            target_nodes=["B", "C", "B"],
            starting_times=[0.0, 1.0, 5.0],
        )

    # --- R1: the last pulse must not be dropped --------------------------- #

    def test_one_laplacian_per_pulse(self, pulse_network):
        """3 pulses -> 3 Laplacian steps (currently only 2 are computed)."""
        pulse_network.compute_laplacian_matrices()
        assert len(pulse_network.laplacians) == 3

    def test_last_pulse_laplacian_reflects_its_event(self, pulse_network):
        """The step for t=5 must couple A and B (pulse (A,B) at t=5)."""
        pulse_network.compute_laplacian_matrices()
        L_last = to_dense(pulse_network.laplacians[-1])
        # nodes: A->0, B->1, C->2; pulse (A,B): rw laplacian couples 0 and 1
        expected = np.array([
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(L_last, expected)

    # --- R2: static adjacency must count pulses, not sum zero durations --- #

    def test_static_adjacency_counts_events(self, pulse_network):
        """Full-range aggregation: (A,B) twice, (B,C) once.

        With zero-duration events the parent's duration-sum aggregation
        yields an all-zero matrix; the meaningful aggregation for pulses
        is the event count per node pair.
        """
        A = pulse_network.compute_static_adjacency_matrix().toarray()
        expected = np.array([
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_allclose(A, expected)

    # --- R3: boundary pulses must count as active -------------------------- #

    def test_default_window_counts_all_pulses(self, pulse_network):
        """The full default window must cover all 3 events and 3 nodes.

        The first pulse lies exactly at t_start = start_time = 0; the
        current mask ``ending_times > t_start`` drops it.
        """
        assert pulse_network.num_active_edges() == 3
        assert pulse_network.num_active_nodes() == 3

    def test_pulse_at_window_start_is_active(self, pulse_network):
        """A pulse exactly at t_start must be included in the window.

        Window [1, 2] contains only the (B,C) pulse at t = 1.
        """
        assert pulse_network.num_active_edges(t_start=1, t_end=2) == 1
        nodes = pulse_network.active_nodes(t_start=1, t_end=2)
        assert sorted(nodes) == [1, 2]  # B and C

    # --- R4: conflicting ending_times must be rejected, not overwritten ---- #

    def test_conflicting_ending_times_column_raises(self):
        """An events_table with nonzero durations must raise ValueError.

        Silently overwriting the user's ending_times hides the mistake of
        feeding an interval network into ContTempInstNetwork.
        """
        df = make_df(
            sources=[0, 1, 2],
            targets=[1, 2, 0],
            starts=[0.0, 1.0, 2.0],
            ends=[10.0, 20.0, 30.0],  # conflicts with zero-duration pulses
        )
        with pytest.raises(ValueError):
            ContTempInstNetwork(events_table=df)

    def test_consistent_ending_times_column_accepted(self):
        """ending_times equal to starting_times is consistent and allowed."""
        df = make_df(
            sources=[0, 1, 2],
            targets=[1, 2, 0],
            starts=[0.0, 1.0, 2.0],
            ends=[0.0, 1.0, 2.0],
        )
        net = ContTempInstNetwork(events_table=df)
        assert net.events_table.ending_times.tolist() == [0.0, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# Solver error paths must fail with meaningful exceptions
# --------------------------------------------------------------------------- #
class TestSolverErrorPaths:
    """Invalid solver arguments must raise ``ValueError``, not crash.

    Two defects in the current implementation:

    * ``_compute_single_T`` validates ``method`` only inside
      ``compute_inter_transition_matrices``; called directly (as
      ``print_report`` and the test suite do) with an unknown method it
      falls through all branches and raises ``UnboundLocalError`` on the
      unbound ``T``.
    * ``mfp_exp`` sets its normality factor ``ai`` only for
      ``non_norm in (0, 1)``; any other value raises ``NameError`` on the
      undefined ``ai`` instead of rejecting the argument.

    Red until both functions validate their inputs explicitly.
    """

    def test_compute_single_T_unknown_method_raises_value_error(self):
        net = ContTempNetwork(
            source_nodes=[0], target_nodes=[1],
            starting_times=[0.0], ending_times=[1.0],
        )
        net.compute_laplacian_matrices()
        with pytest.raises(ValueError):
            net._compute_single_T(
                net.laplacians[0], 1.0, 1.0, net.num_nodes, "no_such_method",
            )

    def test_mfp_exp_invalid_non_norm_raises_value_error(self):
        from scipy.sparse import csr_matrix

        from tempnet.expm_with_tol import mfp_exp

        H = csr_matrix(np.array([[-1.0, 1.0], [1.0, -1.0]]))
        with pytest.raises(ValueError):
            mfp_exp(H, err=1e-8, non_norm=2)


# --------------------------------------------------------------------------- #
# events_table index normalization
# --------------------------------------------------------------------------- #
class TestEventsTableIndexNormalization:
    """The constructor must leave ``events_table`` with a RangeIndex 0..n-1.

    The event index is the join key of the whole pipeline:
    ``_compute_time_grid`` stores it as the ``id`` level of the time grid,
    and the Laplacian loop resolves events with
    ``events_table.loc[id, ...]``. The constructor used to guarantee a
    clean index (``reset_event_table_index`` on main); the integration
    branch dropped that without replacement, so the invariant is assumed
    but never enforced. These tests show three ways it now breaks:

    * lists input with unsorted starting times: ``sort_values`` reorders
      rows but keeps the old index, so the index no longer matches the
      event (= time) order;
    * a DataFrame with duplicate index labels (e.g. from ``pd.concat``):
      ``loc[id]`` resolves to several rows and the (times, id) grid can
      no longer distinguish the events, corrupting the Laplacian loop;
    * a DataFrame with a *named* index: ``reset_index()`` inside
      ``_compute_time_grid`` then produces a column named after the index
      instead of ``"index"``, raising ``KeyError`` far from the cause.
    """

    def test_unsorted_lists_input_gets_reset_index(self):
        net = ContTempNetwork(
            source_nodes=[0, 1], target_nodes=[1, 2],
            starting_times=[5.0, 1.0],  # deliberately unsorted
            ending_times=[6.0, 2.0],
        )
        # rows are sorted by starting time; the index must follow suit
        assert list(net.events_table.index) == [0, 1]

    def test_duplicate_index_dataframe_is_handled(self):
        df = pd.concat([
            make_df([0], [1], starts=[0.0], ends=[1.0]),
            make_df([1], [2], starts=[2.0], ends=[3.0]),
        ])  # index is [0, 0]
        net = ContTempNetwork(events_table=df)
        assert list(net.events_table.index) == [0, 1]

        # end-to-end: the grid [0, 1, 2, 3] must give 3 laplacian steps
        net.compute_laplacian_matrices()
        assert len(net.laplacians) == 3

    def test_named_index_dataframe_is_handled(self):
        df = make_df([0, 1], [1, 2], starts=[0.0, 2.0], ends=[1.0, 3.0])
        df.index.name = "event_id"
        net = ContTempNetwork(events_table=df)

        # must not raise KeyError on the "index" column in the time grid
        net.compute_laplacian_matrices()
        assert len(net.laplacians) == 3
