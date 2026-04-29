"""Legacy shims kept for backwards compatibility.

"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix

from .temporal_network import ContTempNetwork


class ContTempInstNetwork(ContTempNetwork):
    """Continuous time temporal network with instantaneous events.

    This is a subclass of ContTempNetwork for continuous time temporal
    networks where events do not have a duration.

    In practice, it is implemented as a ContTempNetwork where 
    ending_times_k = starting_times_k+1 and where durations (tau_k) = 1  
    for all events for the computation of the transition matrices.

    Parameters
    ----------
    source_nodes: Python list
        List of source nodes of each event, ordered according to
        `starting_times`.

    target_nodes: Python list
        List of target nodes of each event

    starting_times: Python list
        List of starting times of each event

    relabel_nodes: boolean
        Relabel nodes from 0 to num_nodes and save original labels in 
        self.node_to_label_dict. Default is `True`

    reset_event_table_index: boolean
        Reset the index of the `events_table` DataFrame. Default is `True`.

    node_to_label_dict: Python dict
        If `relabel_nodes` is `False, this can be used to save the original labels
        of the nodes.

    events_table: Pandas Dataframe
        Dataframe with columns 'source_nodes', 'target_nodes', 'starting_times'
        and 'ending_times' and index corresponding to event index. Used for
        instantiating a new ConTempNetwork from the event_table of an other one.
    """

    def __init__(self,
                 *,
                 source_nodes=[],
                 target_nodes=[],
                 starting_times=[],
                 relabel_nodes=True,
                 reset_event_table_index=True,
                 node_to_label_dict=None,
                 events_table=None,
                 **kwargs):
        warnings.warn(
            "ContTempInstNetwork is deprecated and will be replaced by a "
            "composition-based API in a future release. It currently still "
            "works unchanged.",
            DeprecationWarning,
            stacklevel=2,
        )

        if events_table is None:
            ending_times = [t + self._DEFAULT_DURATION
                            for t in starting_times]
        else:
            # Instant networks store events_tables without an ending_times
            # column. The parent constructor's events_table branch requires
            # ending_times, so we synthesize it here as start + default
            # duration. For CSV inputs we read the file first, then forward a
            # DataFrame to the parent.
            if isinstance(events_table, (str, Path)):
                events_table = pd.read_csv(str(events_table))
            if isinstance(events_table, pd.DataFrame) and \
                    self._ENDINGS not in events_table.columns:
                events_table = events_table.copy()
                events_table[self._ENDINGS] = (
                    events_table[self._STARTS] + self._DEFAULT_DURATION
                )
            ending_times = []  # ignored when events_table is provided

        super().__init__(source_nodes=source_nodes,
                         target_nodes=target_nodes,
                         starting_times=starting_times,
                         ending_times=ending_times,
                         relabel_nodes=relabel_nodes,
                         reset_event_table_index=reset_event_table_index,
                         node_to_label_dict=node_to_label_dict,
                         merge_overlapping_events=False,
                         events_table=events_table,
                         **kwargs)

        self.events_table["durations"] = [1.0]*self.events_table.shape[0]
        self.instantaneous_events = True

    def compute_laplacian_matrices(self,
                                   *,
                                   t_start=None,
                                   t_stop=None,
                                   save_adjacencies=False):
        """Compute all laplacian matrices and saves them in self.laplacians.

        Computes from the first time index before or equal to t_start until
        the time index before t_stop.

        laplacians are computed from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]

        The laplacian at step k, is the random walk laplacian
        between times[k] and times[k+1]

        NOTE: This subclass implements *pulse dynamics* (state ``A``,
        ``S``, ``Dm1``, ``degrees`` are reset to zero at every time step,
        and event ends are no-ops). This is intentionally distinct from
        ``ContTempNetwork.compute_laplacian_matrices`` which implements
        *interval dynamics* (persistent state across time steps, with
        event ends clearing the corresponding adjacency entry). The
        behavior here mirrors upstream ``TemporalNetwork.py`` at commit
        f99bca3, so the two classes are not expected to produce equal
        laplacians even when ending_times are aligned to start + 1.

        The pulse semantics are encoded entirely via the
        ``_make_adjacency_buffer``, ``_laplacian_prewarm``,
        ``_laplacian_on_event_end`` and ``_laplacian_step_end`` hooks
        below; the loop body itself lives in the parent class.
        """
        return super().compute_laplacian_matrices(
            t_start=t_start,
            t_stop=t_stop,
            save_adjacencies=save_adjacencies,
        )

    # --- pulse-dynamics hook overrides --------------------------------

    def _make_adjacency_buffer(self, n):
        # dok_matrix supports .clear() which is needed in
        # _laplacian_step_end below.
        return dok_matrix((n, n), dtype=np.float64)

    def _laplacian_prewarm(self, state):
        # Pulse dynamics: no events persist across step boundaries, so
        # nothing to seed.
        pass

    def _laplacian_on_event_end(self, state, event):
        # Pulse dynamics: end events are no-ops (state is reset every
        # step in _laplacian_step_end below).
        pass

    def _laplacian_step_end(self, state):
        state.A.clear()
        state.S.data.fill(1.0)
        state.Dm1.data.fill(1.0)
        state.degrees.fill(0.0)

    def compute_inter_transition_matrices(self,
                                          lamda=None,
                                          t_start=None,
                                          t_stop=None,
                                          use_sparse_stoch=False,
                                          dense_expm=True):
        """Compute interevent transition matrices.

        T_k(lamda) = expm(-lamda*L_k).

        Here, for instantaneous events, all events are assumed to have the 
        same duration of unit time (i.e. tau_k =1 for all k).

        The transition matrix T_k is saved in `self.inter_T[lamda][k]`,
        where self.inter_T is a dictionary with lamda as keys and
        lists of transition matrices as values.

        will compute from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]

        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1].
        """
        super().compute_inter_transition_matrices(
            lamda=lamda,
            t_start=t_start,
            t_stop=t_stop,
            fix_tau_k=True,
            use_sparse_stoch=use_sparse_stoch,
            dense_expm=dense_expm
        )

    def compute_lin_inter_transition_matrices(self,
                                              lamda=None,
                                              t_start=None,
                                              t_stop=None,
                                              t_s=10,
                                              use_sparse_stoch=False):
        """Compute interevent transition matrices as a linear approximation.

        Approximation is done for expm(-lamda*L_k) based on the discrete time
        transition matrix.

        Here, for instantaneous events, all events are assumed to have the 
        same duration of unit time (i.e. tau_k =1 for all k).

        `t_s` is the time value for which the linear approximation reaches the
        stationary transition matrix (default is `t_s=10`).

        The transition matrix T_k_lin is saved in 
        `self.inter_T_lin[lamda][t_s][k]`,
        where `self.inter_T_lin` is a dictionary with lamda as keys and
        lists of transition matrices as values.

        will compute from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]

        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        """
        super().compute_lin_inter_transition_matrices(
            lamda=lamda,
            t_start=t_start,
            t_stop=t_stop,
            t_s=t_s,
            fix_tau_k=True,
            use_sparse_stoch=use_sparse_stoch
        )
