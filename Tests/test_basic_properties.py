import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tempnet.temporal_network import ContTempNetwork

@pytest.fixture
def simple_network():
    """
    3-node, 4-event network whose properties are verified by hand.

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
        ending_times=[2, 3, 5, 7],
        relabel_nodes=True,
    )

@pytest.fixture
def mice_network():
    """
    Load mice data 
    """
    url = 'https://zenodo.org/record/4725155/files/'\
                        'mice_contact_sequence.csv.gz'
    cut_after = 24*3600 

    raw_df = pd.read_csv(url,compression='gzip')
    raw_df = raw_df[raw_df['ending_times'] < cut_after]
        
    source_nodes = raw_df['source_nodes'].tolist()
    target_nodes = raw_df['target_nodes'].tolist()
    starting_times = raw_df['starting_times'].round(3).tolist()
    ending_times = raw_df['ending_times'].round(3).tolist()
    return ContTempNetwork(\
        source_nodes=source_nodes,\
        target_nodes=target_nodes,\
        starting_times=starting_times,\
        ending_times=ending_times,\
        relabel_nodes=True
    )

@pytest.fixture
def network_overlapping():
        """Two overlapping events between the same node pair/Three nodes no relabeling."""
        return ContTempNetwork(
            source_nodes=["A","B","A"],
            target_nodes=["B", "C", "B"],
            starting_times=[0, 1, 1],
            ending_times=[3, 2, 4],
            relabel_nodes=True,
            merge_overlapping_events=True
        )
@pytest.fixture
def network_overlapping_no_relabel():
        """Two overlapping events between the same node pair/Three nodes no relabeling."""
        return ContTempNetwork(
            source_nodes=["A","B","A"],
            target_nodes=["B", "C", "B"],
            starting_times=[0, 1, 1],
            ending_times=[3, 2, 4],
            relabel_nodes=False,
            merge_overlapping_events=True
        )
class TestBasicProperties:

  # By Jonas
    def test_init_without_source_nodes(self, mice_network: ContTempNetwork, simple_network: ContTempNetwork):
        for network in [mice_network, simple_network]:
            with pytest.raises(AssertionError):
                ContTempNetwork(target_nodes=network.events_table['target_nodes'].values)

    def test_init_without_target_nodes(self, mice_network: ContTempNetwork, simple_network: ContTempNetwork):
        for network in [mice_network, simple_network]:
            with pytest.raises(AssertionError):
                ContTempNetwork(source_nodes=network.events_table['source_nodes'].values)

    def test_init_inconsistent_event_numbers_type(self):
        """Make sure we detect variable numbers of events
        """
        with pytest.raises(AssertionError):
            # error in source and target nodes
            ContTempNetwork(source_nodes=[1, 2, 3], target_nodes=[1, 2],
                            starting_times = [0, 0], ending_times = [1, 1])
            # not enough ending times
            ContTempNetwork(source_nodes=[1, 2], target_nodes=[1, 2],
                            starting_times = [0, 0], ending_times = [1])

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
                         starting_times = [0, 0], ending_times = [1, 1])
    def test_mismatched_lengths_raise(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(
                source_nodes=["A", "B"],
                target_nodes=["B"],           # wrong length
                starting_times=[0, 1],
                ending_times=[1, 2],
            )

    def test_invalid_events_table_type_raises(self):
        with pytest.raises(ValueError):
            ContTempNetwork(events_table=12345)

    def test_missing_csv_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ContTempNetwork(events_table=Path("nonexistent_file.csv"))

    def test_extra_attrs_wrong_length_raises(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(
                source_nodes=["A"],
                target_nodes=["B"],
                starting_times=[0],
                ending_times=[1],
                extra_attrs={"weight": [1.0, 2.0]},  # too long
            )

    #By yas

    def test_num_nodes(self, simple_network: ContTempNetwork):
        assert simple_network.num_nodes == 3

    def test_num_events(self, simple_network: ContTempNetwork):
        assert simple_network.num_events == 4

    def test_start_time(self, simple_network: ContTempNetwork):
        assert simple_network.start_time == 0

    def test_end_time(self, simple_network: ContTempNetwork):
        assert simple_network.end_time == 7

    def test_node_array_sorted(self, simple_network: ContTempNetwork):
        assert list(simple_network.node_array) == [0, 1, 2]

    def test_durations_column_exists(self, simple_network: ContTempNetwork):
        assert "durations" in simple_network.events_table.columns

    def test_durations_values(self, simple_network: ContTempNetwork):
        expected = [2, 2, 1, 1]
        assert list(simple_network.events_table["durations"]) == expected

    def test_events_sorted_by_start(self, simple_network: ContTempNetwork):
        starts = simple_network.events_table["starting_times"].tolist()
        assert starts == sorted(starts)

    def test_required_columns_present(self, simple_network: ContTempNetwork):
        for col in ["source_nodes", "target_nodes", "starting_times",
                    "ending_times", "durations"]:
            assert col in simple_network.events_table.columns

    def test_index_reset(self, simple_network: ContTempNetwork):
        assert list(simple_network.events_table.index) == list(
            range(simple_network.num_events)
        )

    def test_is_directed_default(self, simple_network: ContTempNetwork):
        assert simple_network.is_directed == False

    def test_instantaneous_events_default(self, simple_network: ContTempNetwork):
        assert simple_network.instantaneous_events == False

    ## TEST adj matrix
    def test_adj_full(self, simple_network: ContTempNetwork):
        A_full=simple_network.compute_static_adjacency_matrix().toarray()
        A_test = np.array([
        [0, 3, 1],
        [3, 0, 2],
        [1, 2, 0]])
        assert np.allclose(A_full, A_test)

    def test_adj_simple1(self , simple_network: ContTempNetwork):
        A=simple_network.compute_static_adjacency_matrix(start_time=0, end_time=2).toarray()
        A_test = np.array([
        [0, 2, 0],
        [2, 0, 1],
        [0, 1, 0]])
        assert np.allclose(A, A_test)

    def test_adj_simple2(self , simple_network: ContTempNetwork):
        A=simple_network.compute_static_adjacency_matrix(start_time=2.5, end_time=3).toarray()
        A_test = np.array([
        [0, 0, 0],
        [0, 0, 0.5],
        [0, 0.5, 0]])
        assert np.allclose(A, A_test)
        

    ############################################################## MICE
    
    def test_num_node_real(self, mice_network: ContTempNetwork):
        n_array=np.load('Tests/Prepare Real test/mice_node_array.npy')
        assert mice_network.num_nodes == len(n_array)

    def test_node_array_real(self, mice_network: ContTempNetwork):
        n_array=np.load('Tests/Prepare Real test/mice_node_array.npy')
        assert sorted(mice_network.node_array) ==sorted(n_array)

    def test_event_table_real(self, mice_network: ContTempNetwork):
        et=pd.read_csv('Tests/Prepare Real test/mice_event_table.csv')
        pd.testing.assert_frame_equal(
            mice_network.events_table.reset_index(drop=True),
            et.reset_index(drop=True)
        )
    def test_compute_time_grid_real(self, mice_network: ContTempNetwork):
        mice_network._compute_time_grid()
        tg=pd.read_csv('Tests/Prepare Real test/mice_time_grid.csv')
        times=np.load('Tests/Prepare Real test/mice_times.npy')

        pd.testing.assert_frame_equal(
                    mice_network.time_grid.reset_index(),
                    tg.reset_index(drop=True)
                )        
        assert list(times)==list(mice_network.times)


    ## TEST adj matrix
    def test_adj_full_mice(self, mice_network: ContTempNetwork):
        A=mice_network.compute_static_adjacency_matrix().toarray()
        A_loaded=np.load('Tests/Prepare Real test/mice_full_adjacency.npy')
        assert np.allclose(A, A_loaded)
        
    def test_adj_1h_mice(self, mice_network: ContTempNetwork):
        A = mice_network.compute_static_adjacency_matrix(
            start_time=0, end_time=3600
        ).toarray()

        A_loaded = np.load('Tests/Prepare Real test/mice_1h_adjacency.npy')

        assert np.allclose(A, A_loaded)
            
   #Overlapping
    def test_overlapping_events_are_merged(self, network_overlapping: ContTempNetwork):
        """(A,B,0,3) and (A,B,1,4) should collapse into a single event."""
        assert network_overlapping.num_events == 2

    def test_merged_event_span(self, network_overlapping: ContTempNetwork):
        row = network_overlapping.events_table.iloc[0]
        assert row["starting_times"] == 0
        assert row["ending_times"] == 4

    def test_merge_flag_set(self, network_overlapping: ContTempNetwork):
        assert network_overlapping._overlapping_events_merged is True


    def test_overlapping_events_are_merged_no_relabel(self, network_overlapping_no_relabel: ContTempNetwork):
        """(A,B,0,3) and (A,B,1,4) should collapse into a single event."""
        assert network_overlapping.num_events == 2

    def test_merged_event_span_no_relabel(self, network_overlapping_no_relabel: ContTempNetwork):
        row = network_overlapping.events_table.iloc[0]
        assert row["starting_times"] == 0
        assert row["ending_times"] == 4

    def test_merge_flag_set_no_relabel(self, network_overlapping_no_relabel: ContTempNetwork):
        assert network_overlapping_no_relabel._overlapping_events_merged is True

