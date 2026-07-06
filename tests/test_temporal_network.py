import os
import pickle
import numpy as np
import pandas as pd
import pytest
from tempnet.utils import to_dense

from tempnet.temporal_network import ContTempNetwork, ContTempInstNetwork


MICE_URL = (
    "https://zenodo.org/record/4725155/files/mice_contact_sequence.csv.gz"
)


MICE_FIXTURE_DIR = "tests/prepare_mice_test"


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
# Real-data tests (mice dataset)
# --------------------------------------------------------------------------- #
class TestMice:

    @staticmethod
    def _fixture(name):
        return os.path.join(MICE_FIXTURE_DIR, name)

    def test_num_nodes(self, mice_network):
        n_array = np.load(self._fixture("mice_node_array.npy"))
        assert mice_network.num_nodes == len(n_array)

    def test_node_array(self, mice_network):
        n_array = np.load(self._fixture("mice_node_array.npy"))
        assert sorted(mice_network.node_array) == sorted(n_array)

    def test_event_table(self, mice_network):
        et = pd.read_csv(self._fixture("mice_event_table.csv"))
        pd.testing.assert_frame_equal(
            mice_network.events_table.reset_index(drop=True),
            et.reset_index(drop=True),
        )

    def test_compute_time_grid(self, mice_network):
        mice_network._compute_time_grid()
        tg = pd.read_csv(self._fixture("mice_time_grid.csv"))
        times = np.load(self._fixture("mice_times.npy"))
        pd.testing.assert_frame_equal(
            mice_network.time_grid.reset_index(),
            tg.reset_index(drop=True),
        )
        assert list(times) == list(mice_network.times)

    def test_adj_full(self, mice_network):
        A = mice_network.compute_static_adjacency_matrix().toarray()
        A_loaded = np.load(self._fixture("mice_full_adjacency.npy"))
        assert np.allclose(A, A_loaded)

    def test_adj_1h(self, mice_network):
        A = mice_network.compute_static_adjacency_matrix(
            start_time=0, end_time=3600,
        ).toarray()
        A_loaded = np.load(self._fixture("mice_1h_adjacency.npy"))
        assert np.allclose(A, A_loaded)

