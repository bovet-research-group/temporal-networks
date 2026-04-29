import pytest

import os
import pickle
import tempfile

from types import SimpleNamespace
from copy import copy

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


from tempnet.temporal_network import ContTempNetwork, ContTempInstNetwork


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


def test_lin_approx_trans_matrix():
    """
    """
    from tempnet.temporal_network import lin_approx_trans_matrix
    pass


def test_compute_stationary_transition():
    """
    """
    from tempnet.temporal_network import compute_stationary_transition
    pass


def test_compute_subspace_expm():
    """
    """
    from tempnet.temporal_network import compute_subspace_expm
    pass


def test_csc_row_normalize():
    """
    """
    from tempnet.temporal_network import csc_row_normalize
    pass


def test_find_spectral_gap():
    """
    """
    from tempnet.temporal_network import find_spectral_gap
    pass


def test_remove_nnz_rowcol():
    """
    """
    from tempnet.temporal_network import remove_nnz_rowcol
    pass


def test_rebuild_nnz_rowcol():
    """
    """
    from tempnet.temporal_network import numpy_rebuild_nnz_rowcol
    pass


def test_sparse_lapl_expm():
    """
    """
    from tempnet.temporal_network import sparse_lapl_expm
    pass


def test_sparse_lin_approx():
    """
    """
    from tempnet.temporal_network import sparse_lin_approx
    pass


def test_sparse_stationary_trans():
    """
    """
    from tempnet.temporal_network import sparse_stationary_trans
    pass


def test_set_to_ones():
    """
    """
    from tempnet.temporal_network import set_to_ones
    pass


def test_set_to_zeroes():
    """
    """
    from tempnet.temporal_network import set_to_zeroes
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


class TestContTempNetworkEndingTimesRequired:
    """ContTempNetwork must raise ValueError when ending_times is absent.

    The validation pinpoints the missing input ('ending_times' or the
    DataFrame column) and directs users to ContTempInstNetwork for
    instantaneous networks.
    """

    def test_positional_no_ending_raises(self):
        with pytest.raises(ValueError) as exc:
            ContTempNetwork(
                source_nodes=[0, 1],
                target_nodes=[1, 0],
                starting_times=[0.0, 1.0],
                ending_times=None,
            )
        msg = str(exc.value)
        assert "ending_times" in msg
        assert "ContTempInstNetwork" in msg

    def test_positional_empty_ending_raises(self):
        with pytest.raises(ValueError) as exc:
            ContTempNetwork(
                source_nodes=[0, 1],
                target_nodes=[1, 0],
                starting_times=[0.0, 1.0],
                ending_times=[],
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

    def test_all_empty_inputs_still_assertion(self):
        """Degenerate empty case must remain an AssertionError, not the
        new ValueError (the validator only fires when starts are present).
        """
        # All-empty lists: existing length-mismatch assertion path is fine,
        # length 0 == 0 == 0 == 0, so this should NOT raise at all.
        net = ContTempNetwork(
            source_nodes=[],
            target_nodes=[],
            starting_times=[],
            ending_times=[],
        )
        assert net.num_events == 0

