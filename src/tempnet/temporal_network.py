"""
#
# Temporal networks `tempnet`
#
# Copyright (C) 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>
#
# Contributors: 
# Yasaman Asgari <yasaman.asgari@uzh.ch>
# Juni Schindler <juni.schindler@uzh.ch>
# Samuel Koovely <samuel.koovely@uzh.ch>

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
import gzip
import os
import pickle
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import gc
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    diags,
    dok_matrix,
    eye,
    isspmatrix_csr,
    lil_matrix,
)

from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm
from .expm_with_tol import mfp_exp
from .subspace_expm import sparse_lapl_expm
from .utils import (
    SparseStochMat,
    inplace_csr_row_normalize,
    set_to_zeroes,
    to_dense,
)

from .logger import get_logger


from matplotlib import pyplot as plt
import seaborn as sns

# get the logger
logger = get_logger()


@dataclass
class _LaplacianState:
    """Mutable buffers shared by ``compute_laplacian_matrices`` and its
    dynamics-specific hooks (``_laplacian_prewarm``,
    ``_laplacian_on_event_end``, ``_laplacian_step_end``).
    """
    A: object        # adjacency buffer (lil_matrix or dok_matrix)
    S: object        # self-loop diagonal (csc)
    Dm1: object      # inverse-degree diagonal (csc)
    degrees: np.ndarray
    dynamics: str


class ContTempNetwork:
    """Continuous time temporal network

    Parameters
    ----------
    source_nodes: Python list
        List of source nodes of each event, ordered according to
        `starting_times`.

    target_nodes: Python list
        List of target nodes of each event.

    starting_times: Python list
        List of starting times of each event.

    ending_times: Python list
        List of ending times of each event.

    extra_attrs: Dict
        Extra event attributes. Must be given in a dict with
        {attr_name: list_of_values}, where list_of_values has the same order
        and length as `source_nodes`.

    label_to_node_dict: Python dict
        The user can input this dictionary to map the labels in an arbitary order. 
    merge_overlapping_events: boolean
        Check for overlapping events (between the same pair of nodes)
        and merges them. Default is `False`.

    events_table: Pandas Dataframe/ URL/Path string to the CSV file
        A Pandas Dataframe or a URL/Path string to
        a CSV file containing the events table o with columns 'source_nodes', 
        'target_nodes', 'starting_times'and 'ending_times' and index 
        corresponding to event index. 
    """
    # parametrize the column names > single place to change them:
    _SOURCES = "source_nodes"
    _TARGETS = "target_nodes"
    _STARTS = "starting_times"
    _ENDINGS = "ending_times"
    _MANDATORY = [_SOURCES, _TARGETS, _STARTS]
    _ESSENTIAL = [_SOURCES, _TARGETS, _STARTS, _ENDINGS]
    # to hold endings - starts
    _DURATIONS = "durations"

    def __init__(self, *,
                 source_nodes=[],
                 target_nodes=[],
                 starting_times=[],
                 ending_times=[],
                 extra_attrs=None,
                 label_to_node_dict=None,
                 merge_overlapping_events=False,
                 events_table=None,
                 **kwargs):

        if events_table is None:
            if (ending_times is None or len(ending_times) == 0) \
                    and len(starting_times) > 0:
                raise ValueError(
                    "ContTempNetwork requires 'ending_times' for each event."
                    " For instantaneous temporal networks use"
                    " ContTempInstNetwork."
                )
            assert len(source_nodes) == len(target_nodes) == \
                   len(starting_times) == len(ending_times)

            data = {"source_nodes": source_nodes,
                    "target_nodes": target_nodes,
                    "starting_times": starting_times,
                    "ending_times": ending_times}
            columns = ["source_nodes", "target_nodes",
                       "starting_times", "ending_times"]

            if extra_attrs is not None:
                assert isinstance(extra_attrs, dict)

                for attr_name, val_list in extra_attrs.items():
                    assert len(val_list) == len(source_nodes)
                    data[attr_name] = val_list
                    columns.append(attr_name)

            self.events_table = pd.DataFrame(data=data,
                                             columns=columns)

            self.events_table.sort_values(by=["starting_times", "ending_times"],
                                          inplace=True)

        else:

            if isinstance(events_table, (str, Path)):
                try:
                    # Convert Path to string if it's a Path object
                    self.events_table = pd.read_csv(str(events_table), **kwargs)
                    logger.debug("Loading events from csv file.")

                except FileNotFoundError:
                    raise ValueError(
                        f"The file at {events_table} was not found."
                    )
                except pd.errors.EmptyDataError:
                    raise ValueError(
                        f"The file at {events_table} is empty."
                    )
                except pd.errors.ParserError:
                    raise ValueError(
                        f"The file at {events_table} could not be parsed."
                    )
                
            elif isinstance(events_table, pd.DataFrame):
                # copy to avoid mutating caller's DataFrame when relabeling
                self.events_table = events_table.copy()

            else:
                raise ValueError(
                    "`events_table` must be a pandas DataFrame or path to CSV file. "
                    f"'{type(events_table)} is not acceptable."
                )

            if self._ENDINGS not in self.events_table.columns:
                raise ValueError(
                    f"events_table is missing required column"
                    f" '{self._ENDINGS}'. For instantaneous temporal"
                    " networks use ContTempInstNetwork."
            )
            if not set(self._ESSENTIAL).issubset(self.events_table.columns):
                    raise ValueError(
                        f"events_table is missing required columns. "
                        f"Expected: {self._ESSENTIAL}, "
                            f"Got: {list(self.events_table.columns)}"
                        )
        self.num_nodes = pd.unique(
            self.events_table[["source_nodes", "target_nodes"]].values.ravel("K")
        ).size  
        if label_to_node_dict: 
            logger.info(label_to_node_dict)  
            values = list(label_to_node_dict.values())
            if len(set(values)) != len(values):
                raise ValueError(
                    "label_to_node_dict must have unique values for each label."
                )
            self.label_to_node_dict = label_to_node_dict
            self.node_to_label_dict = {v: k for k, v in label_to_node_dict.items()}
               
            self.events_table[self._SOURCES] = self.events_table[self._SOURCES].map(self.label_to_node_dict)
            self.events_table[self._TARGETS] = self.events_table[self._TARGETS].map(self.label_to_node_dict)

            if not self._is_contiguous(self.events_table[self._SOURCES],
                                    self.events_table[self._TARGETS]):
                raise ValueError(
                    "Nodes not labeled 0..num_nodes-1 after relabeling."
                )

        elif not self._is_contiguous(self.events_table[self._SOURCES],
                                    self.events_table[self._TARGETS]):
            labels = sorted(set(self.events_table[self._SOURCES]) |
                            set(self.events_table[self._TARGETS]))
            self.label_to_node_dict = {name: i for i, name in enumerate(labels)}
            self.node_to_label_dict = {i: name for name, i in self.label_to_node_dict.items()}
            self.events_table[self._SOURCES] = self.events_table[self._SOURCES].map(self.label_to_node_dict)
            self.events_table[self._TARGETS] = self.events_table[self._TARGETS].map(self.label_to_node_dict)
        else:

            self.label_to_node_dict = {i: i for i in range(self.num_nodes)}
            self.node_to_label_dict = {i: i for i in range(self.num_nodes)}

        self.node_array = np.sort(list(self.label_to_node_dict.values()))



        self.num_events = self.events_table.shape[0]

        self.start_time = self.events_table.starting_times.min()

        self.end_time = self.events_table.ending_times.max()
        
        self.events_table[
            "durations"
        ] = self.events_table.ending_times - self.events_table.starting_times

        # to record compute times
        self._compute_times = {}
        self._overlapping_events_merged = False
        if merge_overlapping_events:
            num_merged = 1
            while num_merged != 0:
                num_merged = self._merge_overlapping_events()
            self._overlapping_events_merged = True

        
    def _is_contiguous(self, src, tgt):
        " This functions checks whether the nodes are indexed from 0 to n-1"
        vals = np.unique(np.concatenate([src.to_numpy(), tgt.to_numpy()]))
        num_nodes=len(vals)
        return (vals.dtype.kind in "iu"
                and vals.min() == 0
                and vals.max() == num_nodes - 1
                and len(vals) == num_nodes)

    def __repr__(self):
        return str(self.__class__) + \
              f" with {self.num_nodes} nodes and {self.num_events} events"

    @property
    def nodes(self):
        """Sorted list of original node labels."""
        return sorted(self.label_to_node_dict.keys())

    @staticmethod
    def _validate_dynamics(dynamics):
        if dynamics is None:
            return "rw"
        if dynamics == "random_walk":
            dynamics = "rw"
        if dynamics not in {"rw", "heat"}:
            raise ValueError("dynamics must be 'rw' or 'heat'")
        return dynamics

    def _resolve_dynamics(self, dynamics=None):
        if dynamics is None:
            dynamics = getattr(self, "laplacian_dynamics", "rw")
        return self._validate_dynamics(dynamics)

    def _invalidate_laplacian_dependent_cache(self):
        """Clear flat caches derived from the current Laplacian matrices."""
        for attr in ("inter_T", "T", "delta_inter_T", "S", "S_reverse"):
            if hasattr(self, attr):
                delattr(self, attr)

        if hasattr(self, "_inter_T_info"):
            delattr(self, "_inter_T_info")

        if hasattr(self, "direction"):
            delattr(self, "direction")

    def _invalidate_lambda_cache(self, lamda):
        for attr in ("T", "delta_inter_T", "S", "S_reverse"):
            if hasattr(self, attr):
                getattr(self, attr).pop(lamda, None)

    def _inter_T_matches_request(self, dynamics, lamda, t_start=None,
                                 t_stop=None, fix_tau_k=None):
        dynamics = self._validate_dynamics(dynamics)
        if not hasattr(self, "_inter_T_info"):
            return False

        info = self._inter_T_info.get(lamda)
        if info is None:
            return False

        t_start_req, k_start_req, t_stop_req, k_stop_req = \
            self._requested_laplacian_range(t_start=t_start, t_stop=t_stop)

        if (
            info.get("dynamics") != dynamics
            or
            info.get("t_start") != t_start_req
            or info.get("k_start") != k_start_req
            or info.get("t_stop") != t_stop_req
            or info.get("k_stop") != k_stop_req
        ):
            return False

        if fix_tau_k is not None and info.get("fix_tau_k") != fix_tau_k:
            return False

        return True

    def _requested_laplacian_range(self, t_start=None, t_stop=None):
        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        if t_start is None:
            t_start_req = self.times[0]
            k_start_req = 0
        else:
            t_start_req, k_start_req = self._get_closest_time(t_start)

        if t_stop is None:
            t_stop_req = self.times[-1]
            k_stop_req = len(self.times) - 1
        else:
            t_stop_req, k_stop_req = self._get_closest_time(t_stop)

        return t_start_req, k_start_req, t_stop_req, k_stop_req

    def _laplacians_match_request(self, t_start=None, t_stop=None,
                                  dynamics=None):
        dynamics = self._validate_dynamics(dynamics)
        if (
            not hasattr(self, "laplacians")
            or getattr(self, "laplacian_dynamics", None) != dynamics
        ):
            return False

        t_start_req, k_start_req, t_stop_req, k_stop_req = \
            self._requested_laplacian_range(t_start=t_start, t_stop=t_stop)

        return (
            getattr(self, "_t_start_laplacians", None) == t_start_req
            and getattr(self, "_k_start_laplacians", None) == k_start_req
            and getattr(self, "_t_stop_laplacians", None) == t_stop_req
            and getattr(self, "_k_stop_laplacians", None) == k_stop_req
        )

    def save(self, filename,
             matrices_list=None,
             attributes_list=None):
        """Save graph event_table and matrices as pickle file

        Parameters
        ----------
        filename: str
            Filename where to save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to save.
            The default list is:
                `matrices_list = ['laplacians',
                                  'adjacencies',
                                  'inter_T',
                                  'T',
                                  'delta_inter_T']`
        attributes_list: list of strings
            List of attribute names to save.
            The default list is:
                `attributes_list = ['label_to_node_dict',
                                    'events_table',
                                    'times',
                                    'time_grid',
                                    'num_nodes',
                                    '_compute_times',
                                    '_t_start_laplacians',
                                    '_k_start_laplacians',
                                    '_t_stop_laplacians',
                                    '_k_stop_laplacians',
                                    '_overlapping_events_merged',]`
        """
        save_dict = dict()

        matrices = ["laplacians",
                    "adjacencies",
                    "inter_T",
                    "T",
                    "delta_inter_T"]

        if matrices_list is None:
            matrices_list = matrices

        attributes = ["label_to_node_dict",
                      "node_to_label_dict",
                      "events_table",
                      "times",
                      "time_grid",
                      "num_nodes",
                      "num_events",
                      "_compute_times",
                      "_t_start_laplacians",
                      "_k_start_laplacians",
                      "_t_stop_laplacians",
                      "_k_stop_laplacians",
                      "laplacian_dynamics",
                      "_inter_T_info",
                      "_overlapping_events_merged"]

        if attributes_list is None:
            attributes_list = attributes

        for attr in attributes_list:
            if hasattr(self, attr):
                save_dict[attr] = getattr(self, attr)

        for mat in matrices_list:
            if hasattr(self, mat):
                save_dict[mat] = getattr(self, mat)

        with open(os.path.splitext(filename)[0] + ".pickle", "wb") as fopen:
            pickle.dump(save_dict, fopen)
        logger.info(f'Network has been successfully saved in {filename}')

    @classmethod
    def load(cls, filename,
             matrices_list=None,
             attributes_list=None):
        """Load network from file and returns a ContTempNetwork instance.

        Parameters
        ----------
        filename: str
            Filename where the network is saved save. The extension is
            automatically added.

        matrices_list: list of strings
            List of names of matrices to load.
            The default list is:
                `matrices_list = ['laplacians',
                                  'adjacencies',
                                  'inter_T',
                                  'T',
                                  'delta_inter_T']`
        attributes_list: list of strings
            List of attribute names to load.
            The default list is:
                `attributes_list = ['label_to_node_dict',
                                    'node_to_label_dict'
                                    'events_table',
                                    'times',
                                    'time_grid',
                                    'num_nodes',
                                    '_compute_times',
                                    '_t_start_laplacians',
                                    '_k_start_laplacians',
                                    '_t_stop_laplacians',
                                    '_k_stop_laplacians',
                                    '_overlapping_events_merged',]`

        """
        matrices = ["laplacians",
                    "adjacencies",
                    "inter_T",
                    "T",
                    "delta_inter_T"]

        if matrices_list is None:
            matrices_list = matrices

        attributes = ["label_to_node_dict",
                      'node_to_label_dict',
                      "events_table",
                      "times",
                      "time_grid",
                      "num_nodes",
                      "num_events",
                      "_compute_times",
                      "_t_start_laplacians",
                      "_k_start_laplacians",
                      "_t_stop_laplacians",
                      "_k_stop_laplacians",
                      "laplacian_dynamics",
                      "_inter_T_info",
                      "_overlapping_events_merged"]

        if attributes_list is None:
            attributes_list = attributes
        # all in a pickle file

        graph_dict = pd.read_pickle(os.path.splitext(filename)[0] + ".pickle")

        events_table = graph_dict.pop("events_table")

        net = cls(events_table=events_table)

        for k, val in graph_dict.items():
            if k in matrices_list:
                setattr(net, k, val)

            if k in attributes_list:
                setattr(net, k, val)

        return net

    def save_inter_T(self,
                     filename,
                     lamda=None,
                     round_zeros=True,
                     tol=1e-8,
                     compressed=False,
                     save_delta=False,
                     replace_existing=False):
        """Saves all the inter transition matrices.

        This method saves all matrixes in `self.inter_T[lamda]` in a pickle
        file togheter with a dictionary including parameters:

        - `_k_start_laplacians`
        - `_k_stop_laplacians`
        - `_t_start_laplacians`
        - `_t_stop_laplacians`
        - `_t_stop_laplacians`
        - `times_k_start_to_k_stop + 1`

          given by
          ```
          self.times.values[
              self._k_start_laplacians: self._k_stop_laplacians + 1
          ]
          ```
        `num_nodes` and `_compute_times`.

        if `save_delta` is True, creates delta_inter_T if it is
        not already present and saves it together with
        inter_T[lamda][0] in a pickle file.
        otherwise, saves inter_T[lamda] directly (good if used with
        SparseStochMat instances).

        Parameters
        ----------
            filename: str
            Filename where the data is saved (.pickle or .gz).
        lamda: float or int, optional.
            Use to save to results for a specific lamba. If None, the results
            for all the computed lambdas will be saved. Default is None.
        round_zeros: bool.
            If True, values smaller than tol*max(abs(inter_T_k)) will be set to
            zero to preserve sparsity. Default is True.
        tol: float
            See round_zeros. Default is 1e-8.
        compressed: bool
            Used to compress the file with gzip. Default is False.
        save_delta: bool
            If True, creates delta_inter_T if it is not already present and
            saves it together with inter_T[lamda][0].
            Only the differences between two consecutives inter_Ts are saved in
            order to minimize file size.
            Must not be used if `use_sparse_stoch` was used in
            `compute_inter_transition_matrices`.
        replace_existing: bool
            If True, erase and replace files if they already exists.
            Default is False.

        Returns
        -------
            None

        """
        assert hasattr(self, "inter_T"), f"PID {os.getpid()} : nothing " \
            "saved, compute inter_T first."

        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:
            if ext != ".gz":
                file += ".gz"
        elif ext != ".pickle":
            file += ".pickle"

        skipping = False
        if os.path.exists(file):
            if replace_existing:
                logger.info("PID %s : file %s already exists, replacing it.", os.getpid(), file)
            else:
                logger.info(f"PID {os.getpid()} : file {file} already exists, skipping.")
                skipping = True

        if not skipping:

            save_dict = {}
            save_dict["_k_start_laplacians"] = self._k_start_laplacians
            save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
            save_dict["_t_start_laplacians"] = self._t_start_laplacians
            save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
            save_dict["times_k_start_to_k_stop+1"] = self.times.values[
                self._k_start_laplacians:self._k_stop_laplacians + 1
            ]
            save_dict["num_nodes"] = self.num_nodes
            save_dict["_compute_times"] = self._compute_times
            if hasattr(self, "laplacian_dynamics"):
                save_dict["laplacian_dynamics"] = self.laplacian_dynamics

            if save_delta:
                if lamda is None:
                    lamdas_to_check = self.inter_T.keys()
                else:
                    lamdas_to_check = [lamda]
                for lamda_i in lamdas_to_check:
                    assert not isinstance(
                        self.inter_T[lamda_i][0], SparseStochMat
                    ), "inter_T must not be SparseStochMat"

                if lamda is not None:
                    self._compute_delta_trans_mat(lamda,
                                                  round_zeros=round_zeros,
                                                  tol=tol)
                else:
                    for lamda_i in self.inter_T.keys():
                        self._compute_delta_trans_mat(
                            lamda_i,
                            round_zeros=round_zeros,
                            tol=tol,
                        )

                if hasattr(self, "delta_inter_T"):
                    save_dict["inter_T"] = dict()
                    save_dict["is_delta_trans"] = True

                    if lamda is None:
                        lamdas = self.delta_inter_T.keys()
                    else:
                        lamdas = [lamda]

                    for lamda in lamdas:
                        save_dict["inter_T"][lamda] = dict()
                        save_dict["inter_T"][lamda][
                            "delta_inter_T"] = self.delta_inter_T[lamda]
                        save_dict["inter_T"][lamda][
                            "trans_mat0"] = self.inter_T[lamda][0].copy()
                        if round_zeros:
                            set_to_zeroes(
                                save_dict["inter_T"][lamda]["trans_mat0"],
                                tol=tol
                            )

                text = "delta trans mats"

            else:

                save_dict["inter_T"] = dict()
                save_dict["is_sparse_stoch"] = True

                if lamda is None:
                    lamdas = self.inter_T.keys()
                else:
                    lamdas = [lamda]

                for lamda in lamdas:
                    assert isinstance(
                        self.inter_T[lamda][0], SparseStochMat
                    ), "inter_T needs to be SparseStochMat"

                    save_dict["inter_T"][lamda] = []
                    for interT in self.inter_T[lamda]:
                        if round_zeros:
                            interT.set_to_zeroes(tol)
                        save_dict["inter_T"][lamda].append(interT.to_dict())

                text = "sparse stoch trans mats"

            if compressed:
                logger.info(f"PID {os.getpid()} : saving {text} to {file}")

                with gzip.open(file, "wb", compresslevel=2) as fopen:
                    pickle.dump(save_dict, fopen)
            else:
                logger.info(f"PID {os.getpid()} : saving {text} to {file}")

                with open(file, "wb") as fopen:
                    pickle.dump(save_dict, fopen)

    @staticmethod
    def load_inter_T(filename):
        """
        Loads inter_T from 'filename'.

        The file must have been generated with `save_inter_T`.

        Returns a dictionary with the inter_T restored.
        """
        ext = os.path.splitext(filename)[-1]

        if ext not in [".pickle", ".gz"]:
            # detect extension
            if os.path.exists(filename + ".pickle"):
                ext = ".pickle"
                filename += ".pickle"
            elif os.path.exists(filename + ".gz"):
                ext = ".gz"
                filename += ".gz"
            elif os.path.exists(filename + ".pickle.gz"):
                ext = ".pickle.gz"
                filename += ".pickle.gz"
            else:
                raise FileNotFoundError(filename)

        if ext == ".gz" or ext == ".pickle.gz":
            with gzip.open(filename,
                           "rb") as fopen:
                load_dict = pickle.load(fopen)

        else:
            with open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        return_dict = {
            "_k_start_laplacians": load_dict["_k_start_laplacians"],
            "_k_stop_laplacians": load_dict["_k_stop_laplacians"],
            "_t_start_laplacians": load_dict["_t_start_laplacians"],
            "_t_stop_laplacians": load_dict["_t_stop_laplacians"],
            "num_nodes": load_dict["num_nodes"],
            "times_k_start_to_k_stop+1": load_dict["times_k_start_to_k_stop+1"]
        }
        if "laplacian_dynamics" in load_dict:
            return_dict["laplacian_dynamics"] = load_dict["laplacian_dynamics"]

        # rebuild inter_T from delta_inter_T
        if "inter_T" in load_dict.keys():
            return_dict["inter_T"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["inter_T"].keys():
                    return_dict["inter_T"][lamda] = [
                        SparseStochMat(**mat_dict)
                        for mat_dict in load_dict["inter_T"][lamda]
                    ]

            else:
                for lamda in load_dict["inter_T"].keys():
                    return_dict["inter_T"][lamda] = \
                        [load_dict["inter_T"][lamda]["trans_mat0"]]

                    for dT in load_dict["inter_T"][lamda]["delta_inter_T"]:
                        return_dict["inter_T"][lamda].append(
                            return_dict["inter_T"][lamda][-1] + dT
                        )
        del load_dict
        return return_dict

    def save_T(self,
               filename,
               lamda=None,
               round_zeros=True,
               tol=1e-8,
               compressed=False):
        """
        Save a dict with 'T' as key and net.T as item with other attributes.

        This also works with SparseStochMat.

        It only works if net.T[lamda] is a matrix and not a list of matrices,
        i.e. if compute_transition_matrices was ran without save_intermediate.

        """
        assert hasattr(
            self, "T"
        ), f"PID {os.getpid()} : nothing saved, compute inter_T first."

        save_dict = {}
        save_dict["_k_start_laplacians"] = self._k_start_laplacians
        save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
        save_dict["_t_start_laplacians"] = self._t_start_laplacians
        save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
        save_dict["times_k_start_to_k_stop+1"] = self.times.values[
            self._k_start_laplacians:self._k_stop_laplacians + 1
        ]
        save_dict["num_nodes"] = self.num_nodes
        save_dict["_compute_times"] = self._compute_times
        if hasattr(self, "laplacian_dynamics"):
            save_dict["laplacian_dynamics"] = self.laplacian_dynamics

        save_dict["T"] = dict()

        if lamda is None:
            lamdas = self.T.keys()
        else:
            lamdas = [lamda]

        for lamda in lamdas:
            if isinstance(self.T[lamda], list):
                raise TypeError(
                    "save_T only supports final transition matrices. "
                    "Run compute_transition_matrices with "
                    "save_intermediate=False."
                )

            if isinstance(self.T[lamda], SparseStochMat):
                save_dict["is_sparse_stoch"] = True
                if round_zeros:
                    T = self.T[lamda].copy()
                    T.set_to_zeroes(tol)
                else:
                    T = self.T[lamda]
                save_dict["T"][lamda] = T.to_dict()

                text = "SparseStochMat T"

            elif isspmatrix_csr(self.T[lamda]) :

                if round_zeros:
                    T = self.T[lamda].copy()
                    set_to_zeroes(T, tol)
                else:
                    T = self.T[lamda]
                save_dict["T"][lamda] = T

                text = "csr T"

            else:
                raise TypeError(
                    "T must be csr or SparseStochMat. current type is "
                    f"{type(self.T[lamda])}"
                )

        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:

            if ext != ".gz":
                file += ".gz"
            logger.info(f"PID {os.getpid()} : saving {text} to {file}")


            with gzip.open(file, "wb", compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen)
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != ".pickle":
                file += ".pickle"
            logger.info(f"PID {os.getpid()} : saving {text} to {file}")

            with open(file, "wb") as fopen:
                pickle.dump(save_dict, fopen)

    @staticmethod
    def load_T(filename):
        """Loads T from 'filename' that was saved with save_T.

        Returns a dictionary with the T restored.

        """
        ext = os.path.splitext(filename)[-1]

        if ext not in [".pickle", ".gz"]:
            # detect extension
            if os.path.exists(filename + ".pickle"):
                ext = ".pickle"
                filename += ".pickle"
            elif os.path.exists(filename + ".gz"):
                ext = ".gz"
                filename += ".gz"
            elif os.path.exists(filename + ".pickle.gz"):
                ext = ".pickle.gz"
                filename += ".pickle.gz"
            else:
                raise FileNotFoundError(filename)

        if ext == ".gz" or ext == ".pickle.gz":
            with gzip.open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        else:
            with open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        return_dict = {
            "_k_start_laplacians": load_dict["_k_start_laplacians"],
            "_k_stop_laplacians": load_dict["_k_stop_laplacians"],
            "_t_start_laplacians": load_dict["_t_start_laplacians"],
            "_t_stop_laplacians": load_dict["_t_stop_laplacians"],
            "num_nodes": load_dict["num_nodes"],
            "times_k_start_to_k_stop+1": load_dict["times_k_start_to_k_stop+1"]
        }
        if "laplacian_dynamics" in load_dict:
            return_dict["laplacian_dynamics"] = load_dict["laplacian_dynamics"]

        if "T" in load_dict.keys():
            return_dict["T"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["T"].keys():
                    return_dict["T"][lamda] = \
                        SparseStochMat(**load_dict["T"][lamda])

            else:
                for lamda in load_dict["T"].keys():
                    return_dict["T"][lamda] = load_dict["T"][lamda]

        del load_dict
        return return_dict


    def compute_static_adjacency_matrix(self, start_time=None, end_time=None):
        """Returns the adjacency matrix of the static network built from the
        aggregagted edge activity between `start_time` and `end_time`.

        Parameters
        ----------
        start_time : float or int, optional
            Starting time for the aggregation. The default is None, i.e. the
            start time of the entire temporal network.
        end_time : float or int, optional
            Ending time for the aggregation. The default is None, i.e. the
            end time of the entire temporal network.

        Returns
        -------
        CSR sparse matrix
            Symmetric adjacency matrix, where element ij is equal to the
            aggregated time during which egde ij was active after `start_time`
            and before `end_time`.

        """
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        mask = np.logical_and(self.events_table.starting_times < end_time,
                            self.events_table.ending_times > start_time)

        sub = self.events_table.loc[mask]


        data, rows, cols = [], [], []
        for ev in sub.itertuples():
            data.append(
                min(ev.ending_times, end_time) - max(ev.starting_times, start_time)
            )
            rows.append(ev.source_nodes)
            cols.append(ev.target_nodes)

        A = coo_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))
        return A + A.T


    def _compute_time_grid(self):
        """Create `self.time_grid`, a dataframe with ('times', 'id') as index,
        wre `id` is the index of the corresponding event in `events_table`,
        and column 'is_start' which is True is the ('times', 'id')
        corresponds to a starting event.
        Also creates `self.times`, an array with all the times values.

        """
        self.time_grid = pd.DataFrame(
            columns=["times", "id", "is_start"],
            index=range(self.events_table.shape[0]*2)
        )
        self.time_grid.iloc[:self.events_table.shape[0], [0, 1]] = \
            self.events_table.reset_index()[["starting_times", "index"]]
        self.time_grid["is_start"] = False
        self.time_grid.loc[0:self.events_table.shape[0]-1, "is_start"] = True

        self.time_grid.iloc[self.events_table.shape[0]:, [0, 1]] = \
            self.events_table.reset_index()[["ending_times", "index"]]

        self.time_grid.times = pd.to_numeric(self.time_grid.times)

        self.time_grid.sort_values("times", inplace=True)

        # group events with same times
        self.time_grid.set_index(["times", "id"], inplace=True)

        self.time_grid.sort_index(inplace=True)

        self.times = self.time_grid.index.levels[0]

    def _get_closest_time(self, t):
        """Return closest smaller or equal time in `times` and its index"""
        if not hasattr(self, "times"):
            self._compute_time_grid()

        if t not in self.times:
            # take the largest smaller time
            if t <= self.times[0]:
                t = self.times[0]
            else:
                t = self.times[self.times <= t].max()

        k = int(np.where(self.times == t)[0][0])

        return t, k

    def compute_laplacian_matrices(self,
                                   *,
                                   t_start=None,
                                   t_stop=None,
                                   save_adjacencies=False, 
                                   dynamics='rw'):
        """Computes the laplacian matrices and saves them in `self.laplacians`

            Computes from the first event time (in `self.times`) before or
            equal to `t_start` until the event time index before `t_stop`.

            Laplacians are computed from `self.times[self._k_start_laplacians]`
            until `self.times[self._k_stop_laplacians-1]`.

            The laplacian at step k, is the random walk laplacian
            between `times[k]`. and `times[k+1]`.

        Parameters
        ----------
        t_start : float or int, optional
            The default is None, i.e. starts at the beginning of times.
            The computation starts from the first time index before or equal
            to t_start. The corresponding starting time index is saved in
            `self._k_start_laplacians` and the real starting time is
            `self.times[self._k_start_laplacians]` which is saved in
            `self._t_start_laplacians`.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Default is end of times.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Similarily to `t_start`, the corresponding times are saved in
            `self._k_stop_laplacians` and `self._t_stop_laplacians`.
        save_adjacencies : bool, optional
            Default is False. Use to save adjacency matrices in
            `self.adjacencies`.

        dynamics : str, optional
            The dynamics to compute the laplacian. Default is 'rw'.
            other option is `heat` laplacian.
            If D is the degree matrix and A is the adjacency matrix: 
                Heat=D-A 
                Random walk=I-D^-1*A 

        Returns
        -------
        None.

        """

        dynamics = self._validate_dynamics(dynamics)
        
        logger.info(f"Computing Laplacians using {dynamics} method")
        self._invalidate_laplacian_dependent_cache()


        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        # instantaneous adjacency matrix (subclasses may override the
        # buffer type, e.g. dok_matrix for pulse dynamics)
        A = self._make_adjacency_buffer(self.num_nodes)

        # identity
        I = eye(self.num_nodes,
                dtype=np.float64).tocsc()

        # degree array
        degrees = np.zeros(self.num_nodes, dtype=np.float64)
        # inverse degrees diagonal matrix
        Dm1 = I.copy()
        # self loop matrix
        S = I.copy()

        state = _LaplacianState(
            A=A,
            S=S,
            Dm1=Dm1,
            degrees=degrees,
            dynamics=dynamics,
        )

        self.laplacians = []
        if save_adjacencies:
            self.adjacencies = []

        # set boundary conditions : L_k is the laplacian during t_k and t_k+1
        if t_start is None:
            self._t_start_laplacians = self.times[0]
            self._k_start_laplacians = 0
        else:
            t, k = self._get_closest_time(t_start)
            self._t_start_laplacians = t
            self._k_start_laplacians = k
        if t_stop is None:
            self._t_stop_laplacians = self.times[-1]
            self._k_stop_laplacians = len(self.times)-1
        else:
            t, k = self._get_closest_time(t_stop)
            self._t_stop_laplacians = t
            self._k_stop_laplacians = k

        t0 = time.time()

        # dynamics-specific pre-warm (interval: seed adjacency from events
        # alive just before _k_start_laplacians; pulse: no-op).
        self._laplacian_prewarm(state)

        # time grid for this time range
        time_grid_range = self.time_grid.loc[(
            self.time_grid.index.get_level_values(
                "times") >= self._t_start_laplacians
        ) & (
            self.time_grid.index.get_level_values(
                "times") < self._t_stop_laplacians
        )]
        
        self.time_grid_range_laplacians=time_grid_range

        for k, (tk, time_ev) in enumerate(
                time_grid_range.groupby(level="times")):
            if not k % 1000:
                logger.info(
                    f"{k} over "
                    f"{self._k_stop_laplacians - self._k_start_laplacians}"
                )
                logger.info(f"{time.time()-t0:.2f}s")

            meet_id = time_ev.index.get_level_values("id")
            # starting or ending events
            is_starts = time_ev.is_start.values



            events_k = [self.events_table.loc[
                mid,
                ["source_nodes", "target_nodes"]
            ].astype(np.int64) for mid in meet_id.values]

            # update instantaneous matrices
            for event, is_start in zip(events_k, is_starts):
                # unweighted, undirected
                if is_start:
                    # if they are not already connected (can happen if the
                    # opposite event overlap)
                    if state.A[event.source_nodes, event.target_nodes] != 1:
                        state.A[event.source_nodes, event.target_nodes] = 1
                        state.A[event.target_nodes, event.source_nodes] = 1

                        state.degrees[event.source_nodes] += 1
                        state.degrees[event.target_nodes] += 1
                else:
                    # dynamics-specific end-of-event handling
                    # (interval: clear A entry & decrement degrees;
                    #  pulse: no-op).
                    self._laplacian_on_event_end(state, event)

                # update self loops from current degrees
                self._laplacian_update_self_loops(state, event)

            # Laplacian L(tk)
            Acsc = state.A.tocsc()
            # T_D = Dm1 @ (Acsc + S)
            # L = I - T_D

            if dynamics == 'rw':
                self.laplacians.append(I - state.Dm1 @ (Acsc + state.S))
                
            elif dynamics == 'heat':
                self.laplacians.append((diags(state.degrees) - Acsc).tocsc())
            
            if save_adjacencies:
                self.adjacencies.append(state.A.copy())

            # dynamics-specific end-of-step handling (interval: persist
            # state across steps; pulse: reset A, S, Dm1, degrees).
            self._laplacian_step_end(state)

        t_end = time.time()-t0
        self.laplacian_dynamics = dynamics
        self._compute_times["laplacians"] = t_end
        self._compute_times[f"laplacians_{dynamics}"] = t_end
        logger.info(f"Finished computing laplacians in {t_end:.2f}")

    # ------------------------------------------------------------------
    # Laplacian-loop extension hooks.
    #
    # Default implementations encode *interval* dynamics (events occupy
    # a finite [start, end) interval, state persists across time steps).
    # ``ContTempInstNetwork`` overrides them to encode *pulse* dynamics
    # (events are instantaneous; state is reset every step).
    # ------------------------------------------------------------------

    def _make_adjacency_buffer(self, n):
        """Allocate the mutable adjacency buffer used by the laplacian loop.

        Default is ``lil_matrix`` (interval dynamics). Pulse dynamics
        overrides this with ``dok_matrix`` because it needs ``.clear()``
        in ``_laplacian_step_end``.
        """
        return lil_matrix((n, n), dtype=np.float64)

    def _laplacian_prewarm(self, state):
        """Seed ``state`` with events that are already active at the time
        step just before ``self._k_start_laplacians``.

        Interval default: replays the matching events to populate ``A``,
        ``degrees``, ``S``, ``Dm1``. Pulse override: no-op.
        """
        if self._k_start_laplacians <= 0:
            return

        t_km1 = self.times[self._k_start_laplacians - 1]

        # find events that have started before or at t_k-1
        # and were still occuring at t_k-1
        mask_ini = (
            self.events_table.starting_times <= t_km1
        ) & (
            self.events_table.ending_times > t_km1
        )
        for event in self.events_table.loc[mask_ini][
            ["source_nodes", "target_nodes"]
        ].itertuples():

            if state.A[event.source_nodes, event.target_nodes] != 1:
                state.A[event.source_nodes, event.target_nodes] = 1
                state.A[event.target_nodes, event.source_nodes] = 1

                state.degrees[event.source_nodes] += 1
                state.degrees[event.target_nodes] += 1

            self._laplacian_update_self_loops(state, event)

    def _laplacian_on_event_end(self, state, event):
        """Handle an event whose ``is_start`` flag is ``False``.

        Interval default: clear the corresponding adjacency entry and
        decrement degrees (when still set). Pulse override: no-op
        (events are instantaneous and state is reset each step).
        """
        if state.A[event.source_nodes, event.target_nodes] > 0:
            state.A[event.source_nodes, event.target_nodes] = 0
            state.A[event.target_nodes, event.source_nodes] = 0

            state.degrees[event.source_nodes] -= 1
            state.degrees[event.target_nodes] -= 1

    def _laplacian_update_self_loops(self, state, event):
        """Sync ``state.S`` and ``state.Dm1`` with ``state.degrees`` for
        the two nodes touched by ``event``.
        """
        for node in (event.source_nodes, event.target_nodes):
            if state.degrees[node] == 0:
                state.S.data[node] = 1
                state.Dm1.data[node] = 1
            else:
                state.S.data[node] = 0
                state.Dm1.data[node] = 1 / state.degrees[node]

    def _laplacian_step_end(self, state):
        """Hook called after each time-step's laplacian is appended.

        Interval default: no-op (state persists). Pulse override:
        resets ``A``, ``S``, ``Dm1``, ``degrees`` to identity/zero so
        the next step starts from a clean slate.
        """
        pass


    def _default_lamda(self):
        if not hasattr(self, "time_grid"):
            self._compute_time_grid()
        taus = np.diff(self.times)
        if taus.size == 0:
            return 1.0
        return 1.0 / np.median(taus)

    def _compute_single_T(self, L, tau_k, lamda, num_nodes, method, **kwargs):
        """Compute a single transition matrix T_k = expm(-tau_k * lamda * L)."""
        if L.getnnz() == 0:
            return eye(num_nodes, format="csr")
        if method == "dense_expm":
            return csr_matrix(expm(-tau_k * lamda * L.toarray()))
        if method == "sparse_expm":
            return expm(-tau_k * lamda * L).tocsr()
        if method == "parallel_expm":
            params = dict(
                dense_expm=True,
                nproc=1,
                thresh_ratio=None,
                normalize_rows=True,
            )
            params.update(kwargs)
            return sparse_lapl_expm(L, fact=tau_k * lamda, **params)
        if method == "mfp_exp":
            params = dict(err=1e-8, non_norm=0)
            params.update(kwargs)
            return mfp_exp(-tau_k * lamda * L, **params).tocsr()
        raise ValueError(f"unknown transition method {method!r}")

    def _prepare_transition_matrix(self, Tk, *, force_csr, tol):
        if isinstance(Tk, SparseStochMat):
            if not force_csr:
                raise ValueError(
                    "inter_T is a SparseStochMat, but force_csr is False. "
                    "Set force_csr=True."
                )
            Tk = Tk.tocsr()
        elif force_csr and not isspmatrix_csr(Tk):
            Tk = Tk.tocsr()
        else:
            Tk = Tk.copy()

        if tol is not None:
            set_to_zeroes(Tk, tol)
        inplace_csr_row_normalize(Tk)
        return Tk

    def compute_inter_transition_matrices(self, *, lamda=None, t_start=None,
                                          t_stop=None, fix_tau_k=False,
                                          dynamics=None,
                                          method="dense_expm", n_jobs=1,
                                          **kwargs):
        """
        Compute inter-event transition matrices for a lambda.

        ``dynamics='rw'`` uses the random-walk Laplacian and
        ``dynamics='heat'`` uses the combinatorial heat Laplacian.
        Results are stored in ``self.inter_T[lamda]`` using the currently
        requested dynamics.
        """
        valid_methods = {"dense_expm", "sparse_expm", "mfp_exp",
                         "parallel_expm"}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method!r}"
            )

        dynamics = self._resolve_dynamics(dynamics)

        if lamda is None:
            logger.info("Taking lamda as 1/tau_w with tau_w = median "
                        "inter-event time")
            lamda = self._default_lamda()

        if not hasattr(self, "inter_T"):
            self.inter_T = {}

        if lamda in self.inter_T:
            if self._inter_T_matches_request(
                dynamics,
                lamda,
                t_start=t_start,
                t_stop=t_stop,
                fix_tau_k=fix_tau_k,
            ):
                logger.debug(
                    "Inter-event transition matrices already computed for "
                    f"{dynamics=}, {lamda=}"
                )
                return

            logger.info(
                "Recomputing inter-event transition matrices for "
                f"{dynamics=}, {lamda=} because the requested range changed"
            )
            self.inter_T.pop(lamda, None)
            self._invalidate_lambda_cache(lamda)

        if not self._laplacians_match_request(
            t_start=t_start,
            t_stop=t_stop,
            dynamics=dynamics,
        ):
            self.compute_laplacian_matrices(
                t_start=t_start,
                t_stop=t_stop,
                dynamics=dynamics,
            )
            if not hasattr(self, "inter_T"):
                self.inter_T = {}

        logger.info(
            "Computing inter-event transition matrices for "
            f"{dynamics=}, {lamda=}"
        )
        t0 = time.time()

        n_steps = len(self.laplacians)
        if len(self.times) < self._k_start_laplacians + n_steps + 1:
            raise ValueError(
                "not enough time-grid points for the current laplacian range"
            )
        if fix_tau_k:
            taus = [1.0] * n_steps
        else:
            taus = [
                self.times[self._k_start_laplacians + k + 1]
                - self.times[self._k_start_laplacians + k]
                for k in range(n_steps)
            ]

        desc = f"expm {dynamics} lambda={float(lamda):.2e}"
        if n_jobs == 1:
            T_list = [
                self._compute_single_T(
                    L, tau, lamda, self.num_nodes, method, **kwargs
                )
                for L, tau in tqdm(
                    zip(self.laplacians, taus),
                    total=n_steps,
                    desc=desc,
                )
            ]
        else:
            results_gen = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(self._compute_single_T)(
                    L, tau, lamda, self.num_nodes, method, **kwargs
                )
                for L, tau in zip(self.laplacians, taus)
            )
            T_list = list(tqdm(results_gen, total=n_steps, desc=desc))

        if len(T_list) == 0:
            logger.debug("no events, trans. matrix = identity")
            T_list.append(eye(self.num_nodes, dtype=np.float64, format="csr"))

        self.inter_T[lamda] = T_list

        if not hasattr(self, "_inter_T_info"):
            self._inter_T_info = {}
        self._inter_T_info[lamda] = {
            "dynamics": dynamics,
            "k_start": self._k_start_laplacians,
            "k_stop": self._k_stop_laplacians,
            "t_start": self._t_start_laplacians,
            "t_stop": self._t_stop_laplacians,
            "fix_tau_k": fix_tau_k,
            "method": method,
            "kwargs": dict(kwargs),
        }

        gc.collect()
        t_end = time.time() - t0
        self._compute_times[f"inter_T_{dynamics}_{lamda}"] = t_end
        logger.debug(
            "Finished inter-event transition matrices for "
            f"{dynamics=}, {lamda=} in {t_end:.2f}s"
        )

    def compute_transition_matrices(self,
                                    lamda=None,
                                    dynamics=None,
                                    save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=False,
                                    tol=None):
        """Compute cumulative transition matrices.

        The matrices are stored as ``self.T[lamda]``. The corresponding
        one-step matrices must already exist in ``self.inter_T[lamda]``.
        """
        dynamics = self._resolve_dynamics(dynamics)

        if lamda is None:
            lamda = self._default_lamda()

        if not hasattr(self, "inter_T") or lamda not in self.inter_T:
            raise Exception("Compute inter_T first.")
        if (
            hasattr(self, "_inter_T_info")
            and lamda in self._inter_T_info
            and self._inter_T_info[lamda].get("dynamics") != dynamics
        ):
            raise Exception(
                "Compute inter_T first for "
                f"dynamics={dynamics!r}."
            )

        if not hasattr(self, "T"):
            self.T = {}

        requested_direction = "reverse" if reverse_time else "forward"

        if lamda in self.T:
            existing_direction = getattr(
                self,
                "direction",
                requested_direction,
            )
            if existing_direction != requested_direction:
                raise ValueError(
                    f"reverse_time={reverse_time} implies "
                    f"'{requested_direction}' direction, but cached "
                    f"{dynamics=} transitions for {lamda=} are "
                    f"'{existing_direction}'."
                )
            if not save_intermediate or isinstance(self.T[lamda], list):
                logger.info(
                    "Transition matrices already computed for "
                    f"{dynamics=}, {lamda=}"
                )
                return

        if hasattr(self, "direction") and self.direction != requested_direction:
            raise ValueError(
                f"reverse_time={reverse_time} implies "
                f"'{requested_direction}' direction, but this network is "
                f"already set to '{self.direction}'."
            )

        inter = self.inter_T[lamda]
        n = len(inter)

        if reverse_time:
            k_init, k_range = n - 1, reversed(range(n - 1))
            self.direction = "reverse"
        else:
            k_init, k_range = 0, range(1, n)
            self.direction = "forward"

        logger.info(
            "Computing transition matrices for "
            f"{dynamics=}, lambda={lamda} in {self.direction} time"
        )

        t0 = time.time()
        T0 = self._prepare_transition_matrix(
            inter[k_init],
            force_csr=force_csr,
            tol=tol,
        )

        if save_intermediate:
            self.T[lamda] = [T0]
            for k in k_range:
                if not k % 1000:
                    logger.info(f"{k} over {n} - {time.time() - t0:.2f}s")

                Tk = self._prepare_transition_matrix(
                    inter[k],
                    force_csr=force_csr,
                    tol=tol,
                )
                self.T[lamda].append(self.T[lamda][-1] @ Tk)
                inplace_csr_row_normalize(self.T[lamda][-1])
        else:
            self.T[lamda] = T0
            for k in k_range:
                if not k % 1000:
                    logger.info(f"{k} over {n} - {time.time() - t0:.2f}s")
                Tk = self._prepare_transition_matrix(
                    inter[k],
                    force_csr=force_csr,
                    tol=tol,
                )
                self.T[lamda] = self.T[lamda] @ Tk
                inplace_csr_row_normalize(self.T[lamda])

        self._compute_times[
            f"trans_matrix_{dynamics}_{lamda}_rev{reverse_time}"
        ] = time.time() - t0
        logger.info(
            "Finished computing transition matrices for "
            f"{dynamics=}, lambda={lamda}"
        )

    @staticmethod
    def _conditional_entropy_of_transition_matrix(T, p0):
        """Return H(X_t | X_0) for transition matrix ``T`` and weights ``p0``."""
        if not isspmatrix_csr(T):
            T = T.tocsr()

        p0 = np.asarray(p0, dtype=np.float64)
        data = T.data
        indptr = T.indptr

        if data.size == 0:
            return 0.0

        xlogx = np.zeros_like(data, dtype=np.float64)
        mask = data > 0
        xlogx[mask] = data[mask] * np.log(data[mask])

        row_lengths = np.diff(indptr)
        row_sums = np.zeros(T.shape[0], dtype=np.float64)
        nonempty = row_lengths > 0
        if np.any(nonempty):
            starts = indptr[:-1][nonempty]
            row_sums[nonempty] = np.add.reduceat(xlogx, starts)

        return float(-np.dot(p0, row_sums))

    def _compute_cumulative_transition_sequence(self, lamda, dynamics=None,
                                                reverse_time=False,
                                                force_csr=True, tol=None):
        """Build cumulative transition matrices without changing ``self.T``."""
        if not hasattr(self, "inter_T") or lamda not in self.inter_T:
            raise Exception("Compute inter_T first.")

        inter = self.inter_T[lamda]
        if reverse_time:
            k_init = len(inter) - 1
            k_range = reversed(range(k_init))
        else:
            k_init = 0
            k_range = range(1, len(inter))

        cumulative = self._prepare_transition_matrix(
            inter[k_init],
            force_csr=force_csr,
            tol=tol,
        )
        transition_sequence = [cumulative]

        for k in k_range:
            Tk = self._prepare_transition_matrix(
                inter[k],
                force_csr=force_csr,
                tol=tol,
            )
            cumulative = cumulative @ Tk
            if tol is not None:
                set_to_zeroes(cumulative, tol)
            inplace_csr_row_normalize(cumulative)
            transition_sequence.append(cumulative)

        return transition_sequence

    def compute_entropy(self, lamda=None, dynamics=None, t_start=None,
                        t_stop=None, method="dense_expm", n_jobs=1,
                        fix_tau_k=False, verbose=False, reverse_time=False,
                        force_csr=True, tol=None, alpha_sampling=None,
                        **kwargs):
        """Compute global conditional entropy of cumulative transitions.

        Entropy is cached as ``self.S[lamda]`` for forward time and
        ``self.S_reverse[lamda]`` for reverse time.
        """
        if not force_csr:
            raise Exception("Use force_csr=True")

        dynamics = self._resolve_dynamics(dynamics)

        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        if lamda is None:
            if verbose:
                logger.info("Taking lamda as 1/tau_w with tau_w = median "
                            "inter-event time")
            lamda = self._default_lamda()

        if (
            not hasattr(self, "inter_T")
            or lamda not in self.inter_T
            or not self._inter_T_matches_request(
                dynamics,
                lamda,
                t_start=t_start,
                t_stop=t_stop,
                fix_tau_k=fix_tau_k,
            )
        ):
            self.compute_inter_transition_matrices(
                lamda=lamda,
                t_start=t_start,
                t_stop=t_stop,
                fix_tau_k=fix_tau_k,
                dynamics=dynamics,
                method=method,
                n_jobs=n_jobs,
                **kwargs,
            )

        if not hasattr(self, "S"):
            self.S = {}

        if reverse_time and not hasattr(self, "S_reverse"):
            self.S_reverse = {}

        entropy_cache = self.S_reverse if reverse_time else self.S

        if alpha_sampling is None and lamda in entropy_cache:
            entropy_values = np.asarray(entropy_cache[lamda],
                                        dtype=np.float64)
            sampled_indices = np.arange(entropy_values.size, dtype=int)
            if reverse_time:
                sampled_indices = sampled_indices[::-1]
            return np.column_stack((
                sampled_indices.astype(np.float64),
                entropy_values,
            ))

        transition_matrices = None
        if not reverse_time:
            if (
                not hasattr(self, "T")
                or lamda not in self.T
                or not isinstance(self.T[lamda], list)
            ):
                self.compute_transition_matrices(
                    lamda=lamda,
                    dynamics=dynamics,
                    save_intermediate=True,
                    reverse_time=reverse_time,
                    force_csr=force_csr,
                    tol=tol,
                )

            if (
                hasattr(self, "T")
                and lamda in self.T
                and isinstance(self.T[lamda], list)
            ):
                transition_matrices = self.T[lamda]

        if transition_matrices is None:
            transition_matrices = self._compute_cumulative_transition_sequence(
                lamda,
                dynamics=dynamics,
                reverse_time=reverse_time,
                force_csr=force_csr,
                tol=tol,
            )

        num_points = len(transition_matrices)
        all_indices = np.arange(num_points, dtype=int)

        if alpha_sampling is None:
            sampled_indices = all_indices
        else:
            alpha = float(alpha_sampling)
            if alpha <= 0 or alpha > 1:
                raise ValueError("alpha_sampling must be in (0, 1].")

            num_samples = max(1, int(np.ceil(alpha * num_points)))
            if num_samples >= num_points:
                sampled_indices = all_indices
            else:
                inter_info = getattr(self, "_inter_T_info", {}).get(lamda, {})
                time_start = inter_info.get(
                    "k_start",
                    getattr(self, "_k_start_laplacians", 0),
                )
                time_stop = time_start + num_points
                if hasattr(self, "times") and len(self.times) >= time_stop:
                    candidate_times = np.asarray(
                        self.times[time_start:time_stop],
                        dtype=np.float64,
                    )
                    target_times = np.linspace(
                        float(candidate_times[0]),
                        float(candidate_times[-1]),
                        num_samples,
                    )
                    sampled_indices = np.searchsorted(
                        candidate_times,
                        target_times,
                        side="left",
                    )
                    sampled_indices = np.clip(
                        sampled_indices,
                        0,
                        num_points - 1,
                    )
                    sampled_indices = np.unique(sampled_indices).astype(int)
                else:
                    sampled_indices = np.unique(
                        np.round(
                            np.linspace(0, num_points - 1, num_samples)
                        ).astype(int)
                    )

        if reverse_time:
            sampled_indices = sampled_indices[::-1]

        if num_points == 0:
            entropy_signal = np.empty((0, 2), dtype=np.float64)
            if alpha_sampling is None:
                entropy_cache[lamda] = []
            return entropy_signal

        p0 = np.full(self.num_nodes, 1 / self.num_nodes, dtype=np.float64)

        if verbose:
            if reverse_time:
                logger.info("Reversed time computation.")
            logger.info("Computing entropy")

        t0 = time.time()
        entropy_values = np.empty(len(sampled_indices), dtype=np.float64)
        for pos, k in enumerate(sampled_indices):
            if verbose and not pos % 1000:
                logger.info(f"{pos} over {len(sampled_indices)}")
                logger.info(f"{time.time()-t0:.2f}s")

            entropy_values[pos] = \
                self._conditional_entropy_of_transition_matrix(
                    transition_matrices[k],
                    p0,
                )

        t_end = time.time() - t0

        if alpha_sampling is None:
            entropy_cache[lamda] = entropy_values.tolist()
            self._compute_times[
                f"entropy_{dynamics}_{lamda}_rev{reverse_time}"
            ] = t_end
        else:
            self._compute_times[
                f"entropy_{dynamics}_{lamda}_rev{reverse_time}"
                f"_alpha{alpha_sampling}"
            ] = t_end

        if verbose:
            logger.info(f"Finished computing entropy in {t_end:.2f}s")

        return np.column_stack((
            sampled_indices.astype(np.float64),
            entropy_values,
        ))

    def compute_entropy_upper_bound(self, t_start=None, t_stop=None,
                                    aggregation_start_time=None,
                                    alpha_sampling=None, return_times=False,
                                    verbose=False):
        """Compute a component-size upper bound for the entropy curve.

        For each sampled transition interval, the bound is computed from the
        connected components of the static graph aggregated from
        ``aggregation_start_time`` to that interval's end time:
        ``sum_c (|c| / N) log(|c|)``.

        Parameters
        ----------
        t_start : float or int, optional
            Starting time used to select the sample range. The default is the
            first time in ``self.times``.
        t_stop : float or int, optional
            Stopping time used to select the sample range. The default is the
            last time in ``self.times``.
        aggregation_start_time : float or int, optional
            Start time for the cumulative static graph. The default is the
            selected ``t_start``.
        alpha_sampling : float, optional
            Fraction of samples to evaluate, in ``(0, 1]``.
        return_times : bool, optional
            If ``False`` (default), the first returned column contains sample
            indices. If ``True``, it contains interval end times.
        verbose : bool, optional
            Log progress information.

        Returns
        -------
        numpy.ndarray
            Two-column array with sample indices or times in the first column
            and upper-bound values in the second column.
        """
        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        if self.num_nodes == 0:
            return np.empty((0, 2), dtype=np.float64)

        if t_start is None:
            selected_t_start = self.times[0]
            k_start = 0
        else:
            selected_t_start, k_start = self._get_closest_time(t_start)

        if t_stop is None:
            k_stop = len(self.times) - 1
        else:
            _, k_stop = self._get_closest_time(t_stop)

        num_points = max(0, k_stop - k_start)
        if num_points == 0:
            return np.empty((0, 2), dtype=np.float64)

        if aggregation_start_time is None:
            aggregation_start_time = selected_t_start

        all_indices = np.arange(num_points, dtype=int)
        if alpha_sampling is None:
            sampled_indices = all_indices
        else:
            alpha = float(alpha_sampling)
            if alpha <= 0 or alpha > 1:
                raise ValueError("alpha_sampling must be in (0, 1].")

            num_samples = max(1, int(np.ceil(alpha * num_points)))
            if num_samples >= num_points:
                sampled_indices = all_indices
            else:
                candidate_times = np.asarray(
                    self.times[k_start:k_stop],
                    dtype=np.float64,
                )
                target_times = np.linspace(
                    float(candidate_times[0]),
                    float(candidate_times[-1]),
                    num_samples,
                )
                sampled_indices = np.searchsorted(
                    candidate_times,
                    target_times,
                    side="left",
                )
                sampled_indices = np.clip(
                    sampled_indices,
                    0,
                    num_points - 1,
                )
                sampled_indices = np.unique(sampled_indices).astype(int)

        sample_times = np.asarray(
            self.times[k_start + sampled_indices + 1],
            dtype=np.float64,
        )
        values = np.empty(len(sampled_indices), dtype=np.float64)

        if verbose:
            logger.info("Computing entropy upper bound")
        t0 = time.time()

        for pos, end_time in enumerate(sample_times):
            if verbose and not pos % 1000:
                logger.info(f"{pos} over {len(sampled_indices)}")
                logger.info(f"{time.time()-t0:.2f}s")

            if float(aggregation_start_time) >= float(end_time):
                adjacency = csr_matrix(
                    (self.num_nodes, self.num_nodes),
                    dtype=np.float64,
                )
            else:
                adjacency = self.compute_static_adjacency_matrix(
                    start_time=float(aggregation_start_time),
                    end_time=float(end_time),
                ).tocsr()
                adjacency.eliminate_zeros()

            n_components, labels = connected_components(
                adjacency,
                directed=False,
                return_labels=True,
            )
            component_sizes = np.bincount(
                labels,
                minlength=n_components,
            ).astype(np.float64)
            weights = component_sizes / float(self.num_nodes)
            values[pos] = float(np.sum(weights * np.log(component_sizes)))

        self._compute_times["entropy_upper_bound"] = time.time() - t0
        if verbose:
            logger.info(
                "Finished computing entropy upper bound in "
                f"{self._compute_times['entropy_upper_bound']:.2f}s"
            )

        x_values = sample_times if return_times else sampled_indices
        return np.column_stack((
            np.asarray(x_values, dtype=np.float64),
            values,
        ))

    def active_nodes(self, t_start=None, t_end=None):

        """Return the nodes that are active within a given time window.

        A node is considered active if it is an endpoint of at least one event
        that overlaps the interval ``[t_start, t_end]``. An event overlaps the
        window when it starts before ``t_end`` and ends after ``t_start``.

        Parameters
        ----------
        t_start : float or int (default is None)
            Start of the time window. Must be strictly less than ``t_end``.
        t_end : float or int (default is None)
            End of the time window.

        Returns
        -------
        numpy.ndarray
            Array of unique node ids active within the window. Empty if no events overlap.
        """

        if not t_start: 
            t_start=self.start_time
        if not t_end: 
            t_end=self.end_time

        assert t_start < t_end, \
            "t_end should be bigger than t_start"

        t_start=max(self.start_time, t_start)
        t_end=min(self.end_time, t_end)    
        mask = (self.events_table["starting_times"] < t_end) & (self.events_table["ending_times"] > t_start)
        edges = self.events_table[mask]
        nodes = set(edges["source_nodes"]).union(set(edges["target_nodes"]))
        return np.sort(np.array(list(nodes)))

    def num_active_nodes(self, t_start=None, t_end=None):
        """Return the number of nodes active within a given time window.

        A node is active if it is an endpoint of at least one event overlapping
        ``[t_start, t_end]``. 

        Parameters
        ----------
        t_start : float or int (default: None)
            Start of the time window. Must be strictly less than ``t_end``.
        t_end : float or int (default: None)
            End of the time window.

        Returns
        -------
        int
            Number of active nodes in the window. Zero if no events
            overlap.
        """        
        nodes=self.active_nodes(t_start, t_end)
        return len(nodes)


    def num_active_edges(self, t_start=None, t_end=None):
        """Return the number of edges active within a given time window.

        An edge (event) is counted as active if it overlaps the interval
        ``[t_start, t_end]``, that is, it starts before ``t_end`` and ends
        after ``t_start``. 

        Note that this counts *events*, so if the same node pair interacts
        multiple times within the window, each interaction is counted
        separately.

        Parameters
        ----------
        t_start : float or int
            Start of the time window. Must be strictly less than ``t_end``.
        t_end : float or int
            End of the time window.

        Returns
        -------
        int
            Number of active events overlapping the window. Zero if none.
        """
        if not t_start: 
            t_start=self.start_time
        if not t_end: 
            t_end=self.end_time

        assert t_start < t_end, \
            "t_end should be bigger than t_start"

        t_start = max(self.start_time, t_start)
        t_end = min(self.end_time, t_end)

        mask = (self.events_table["starting_times"] < t_end) & \
               (self.events_table["ending_times"] > t_start)
        return int(mask.sum())

    
    def plot_density_of_laplacians(self):
        """Plot the distribution of Laplacian densities.

        For each Laplacian ``L`` in ``self.laplacians``, the density is computed
        as the number of stored (non-zero) entries divided by ``num_nodes ** 2``.
        The densities are shown as a histogram on log-log axes.

        The method also find and plots the time slice corresponding to the 0th (min), 25th, 50th (median), 75th, and
        100th (max) percentiles of the density distribution, and returns their indices. These
        can be used to choose the fastest method for computing transition matrices.
        Returns
        -------
        list of int
            Indices into ``self.laplacians`` of the slices closest to the
            0th, 25th, 50th, 75th, and 100th percentiles of density, in that
            order.
        """

        # density per slice: nnz normalized by N^2
        density = np.array([L.nnz / (self.num_nodes ** 2) for L in self.laplacians])

        # quantile values and the indices of the slices closest to them
        quantiles = np.quantile(density, [0, 0.25, 0.50, 0.75, 1])
        indices = [np.argmin(np.abs(density - q)) for q in quantiles]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)
        sns.histplot(density, ax=ax, bins=np.logspace(-5, 0, 21),
                    fill=False, element='step')
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel('Density of Laplacians')
        ax.set_ylabel('Count')

        # interquartile range (25th–75th) shaded
        ax.axvspan(density[indices[1]], density[indices[3]], alpha=0.1, color='r')
        ax.axvline(density[indices[1]], color='r', linestyle='--', linewidth=0.5)
        ax.axvline(density[indices[3]], color='r', linestyle='--', linewidth=0.5)
        ax.axvline(density[indices[2]], color='r', linestyle='--', label='Median')

        # min and max
        ax.axvline(density[indices[0]], color='b', linestyle=':', linewidth=1, label='Min/Max')
        ax.axvline(density[indices[-1]], color='b', linestyle=':', linewidth=1)

        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        return indices

    def print_report(self, indices, scales, method_kwargs=None, **kwargs):
        """Benchmark and compare matrix-exponential computation methods.

        Args:
            indices: Iterable of 5 integer indices into ``self.laplacians`` /
                ``self.times``, mapped to the labels
                ['min', 'q25', 'median', 'q75', 'max'].
            scales: Iterable of diffusion scale factors (``lamda``) to sweep.
            method_kwargs: Optional dict mapping a method name to a dict of
                extra keyword args for that method, e.g.
                ``{'mfp_exp': {'err': 1e-6}, 'parallel_expm': {'nproc': 4}}``.
        """
        method_kwargs = method_kwargs or {}
        labels = ['min', 'q25', 'median', 'q75', 'max']
        laplacians = {
            label: {
                'L': self.laplacians[idx],
                'tau': self.times[idx + 1] - self.times[idx],
            }
            for label, idx in zip(labels, indices)
        }
        methods = ['dense_expm', 'sparse_expm', 'mfp_exp', 'parallel_expm']
        num_nodes = self.num_nodes
        reference = 'dense_expm'

        scales = list(scales)
        min_scale_idx = int(np.argmin(scales))
        max_scale_idx = int(np.argmax(scales))

        results = {}
        outputs = {}
        for method in methods:
            this_kwargs = {**kwargs, **method_kwargs.get(method, {})}
            for label, data in laplacians.items():
                times = []
                mats = []
                for lamda in scales:
                    t = time.perf_counter()
                    T = self._compute_single_T(
                        L=data['L'], tau_k=data['tau'], lamda=lamda,
                        num_nodes=num_nodes, method=method,
                        **this_kwargs
                    )
                    times.append(time.perf_counter() - t)
                    mats.append(T)
                results[(method, label)] = times
                outputs[(method, label)] = mats

        mfp_mae = {}  # label -> list of per-scale MAE
        for label in laplacians:
            errs = []
            for T_ref, T_approx in zip(
                outputs[(reference, label)], outputs[('mfp_exp', label)]
            ):
                errs.append(
                    np.mean(np.abs(to_dense(T_ref) - to_dense(T_approx)))
                )
            mfp_mae[label] = errs

        # Aggregate MAE across labels, per scale
        mae_avg = np.mean([np.mean(mfp_mae[lbl]) for lbl in laplacians])
        mae_min_scale = np.mean([mfp_mae[lbl][min_scale_idx] for lbl in laplacians])
        mae_max_scale = np.mean([mfp_mae[lbl][max_scale_idx] for lbl in laplacians])

        # Report
        for method in methods:
            print(f"\n=== {method} ===")
            method_total = 0.0
            for label in laplacians:
                times = results[(method, label)]
                method_total += sum(times)
                line = (f"  L_{label:<7} avg={np.mean(times):.4f}s  "
                        f"min_scale={min(times):.4f}s  max_scale={max(times):.4f}s")
                if method == 'mfp_exp':
                    errs = mfp_mae[label]
                    line += (f"  MAE(avg={np.mean(errs):.3e}, "
                            f"min_scale={errs[min_scale_idx]:.3e}, "
                            f"max_scale={errs[max_scale_idx]:.3e})")
                print(line)
            print(f"  total: {method_total:.4f}s")
            if method == 'mfp_exp':
                print(f"  overall MAE vs {reference}:  "
                    f"avg={mae_avg:.3e}  "
                    f"min_scale(={scales[min_scale_idx]:g})={mae_min_scale:.3e}  "
                    f"max_scale(={scales[max_scale_idx]:g})={mae_max_scale:.3e}")

        totals = {
            m: sum(sum(results[(m, lbl)]) for lbl in laplacians)
            for m in methods
        }
        best = min(totals, key=totals.get)
        print(f"\nRecommended method: {best} "
            f"({totals[best]:.4f}s total, fastest of the three)")

    def _compute_delta_trans_mat(self, lamda, round_zeros=True, tol=1e-8):
        """Compute differences between consecutive inter-event matrices."""
        if not hasattr(self, "inter_T") or lamda not in self.inter_T:
            return

        if not hasattr(self, "delta_inter_T"):
            self.delta_inter_T = {}

        if lamda in self.delta_inter_T:
            logger.debug(
                f"delta_inter_T already computed for {lamda=}"
            )
            return

        self.delta_inter_T[lamda] = [
            self.inter_T[lamda][k + 1] - self.inter_T[lamda][k]
            for k in range(len(self.inter_T[lamda]) - 1)
        ]

        if round_zeros:
            for M in self.delta_inter_T[lamda]:
                set_to_zeroes(M, tol=tol)


    def _merge_overlapping_events(self):
        """
        Merge temporally overlapping undir. event between each pair of nodes.
        """
        events_to_keep = np.ones(self.events_table.shape[0], dtype=bool)

        A = self.compute_static_adjacency_matrix()

        # loop over nodes

        for i, n1 in enumerate(self.node_array):
            for j in (A[i, :] > 0).nonzero()[1]:
                n2 = self.node_array[j]
                mask_12 = np.logical_and(
                    self.events_table.source_nodes.values == n1,
                    self.events_table.target_nodes.values == n2
                )

                mask_21 = np.logical_and(
                    self.events_table.source_nodes.values == n2,
                    self.events_table.target_nodes.values == n1
                )
                # sort by starting times
                evs = self.events_table.loc[
                    np.logical_or(mask_12, mask_21)
                ].sort_values(by=["starting_times", "ending_times"])

                evs_list = list(evs.itertuples())

                # event to compare
                ev1 = evs_list[0]
                merged = 0
                for k in range(1, len(evs_list)):
                    ev2 = evs_list[k]
                    # if ev2 overlaps with ev1, merge them
                    # otherwise ev2 becomes ev1
                    if ev2.starting_times < ev1.ending_times:
                        # merge
                        events_to_keep[ev2.Index] = False
                        self.events_table.loc[
                            ev1.Index, "ending_times"] = ev2.ending_times
                        ev1._replace(ending_times=ev2.ending_times)
                        merged += 1
                    else:
                        ev1 = ev2
                if merged !=0: 
                    logger.debug(f"n1,n2 ({n1},{n2}): {merged} merged")

        num_merged = (events_to_keep == False).sum()
        if num_merged !=0: 
            logger.info(f"Merged {num_merged} events.")
        else: 
            logger.debug(f"Merged {num_merged} events.")

        self.events_table = self.events_table.loc[events_to_keep]

        self.events_table.reset_index(inplace=True, drop=True)

        self.num_nodes = self.node_array.shape[0]

        self.num_events = self.events_table.shape[0]

        self.start_time = self.events_table.starting_times.min()

        self.end_time = self.events_table.ending_times.max()

        self._compute_time_grid()

        return num_merged



class ContTempInstNetwork(ContTempNetwork):
    """Continuous time temporal network with instantaneous events.

    This is a subclass of ContTempNetwork for continuous time temporal
    networks where events do not have a duration.

    Parameters
    ----------
    source_nodes: Python list
        List of source nodes of each event, ordered according to
        `starting_times`.

    target_nodes: Python list
        List of target nodes of each event

    starting_times: Python list
        List of starting times of each event

    label_to_node_dict: Python dict
        User may input this to map the nodes in arbitary order.

    events_table: Pandas Dataframe or Url/path to csv file
        Dataframe with columns 'source_nodes', 'target_nodes', 'starting_times'
        and index corresponding to event index. Used for
        instantiating a new ContTempInstNetwork from the event_table of an other one.
    """
    def __init__(self,
                 source_nodes=None,
                 target_nodes=None,
                 starting_times=None,
                 label_to_node_dict=None,
                 events_table=None,
                 ):

        if source_nodes is None:
            source_nodes = []
        if target_nodes is None:
            target_nodes = []
        if starting_times is None:
            starting_times = []

        if events_table is not None:
            if isinstance(events_table, (str, Path)):
                events_table = pd.read_csv(str(events_table))

            if isinstance(events_table, pd.DataFrame):
                if not all(col in events_table.columns
                           for col in ["source_nodes", "target_nodes", "starting_times"]):
                    raise ValueError(
                        "events_table must contain columns 'source_nodes', "
                        "'target_nodes', and 'starting_times'")
                source_nodes = events_table["source_nodes"].values
                target_nodes = events_table["target_nodes"].values
                starting_times = events_table["starting_times"].values
            else:
                raise ValueError(
                    "events_table must be a pandas DataFrame or a path to a csv file")

        super().__init__(source_nodes=source_nodes,
                         target_nodes=target_nodes,
                         starting_times=starting_times,
                         ending_times=starting_times,
                         label_to_node_dict=label_to_node_dict,
                         merge_overlapping_events=False,
                         )
        
        # remove duration column as it doesnt make sense for instantaneous events
        self.events_table.drop(columns=[self._DURATIONS], inplace=True, errors="ignore")

    def compute_laplacian_matrices(self,
                                   *,
                                   t_start=None,
                                   t_stop=None,
                                   save_adjacencies=False,
                                   dynamics="rw"):
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
            dynamics=dynamics,
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


    def compute_inter_transition_matrices(self, *, lamda=None, t_start=None,
                                          t_stop=None, fix_tau_k=True,
                                          dynamics=None,
                                          method="dense_expm", n_jobs=1,
                                          **kwargs):

        """Compute interevent transition matrices.

        T_k(lamda) = expm(-lamda*L_k).

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
            dynamics=dynamics,
            method=method,
            n_jobs=n_jobs,
            **kwargs,
        )
