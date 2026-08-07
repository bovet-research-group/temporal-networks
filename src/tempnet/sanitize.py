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

"""Event-table normalization ("sanitation") for temporal networks.

The temporal-network machinery relies on three invariants of the
``events_table`` DataFrame:

1. Node ids in ``source_nodes`` / ``target_nodes`` are contiguous
   integers ``0..N-1``.
2. Events are sorted chronologically by
   ``(starting_times, ending_times)``.
3. The index is a zero-based ``RangeIndex`` (``0..num_events-1``).

:func:`needs_sanitization` checks whether a DataFrame satisfies all
three; :func:`sanitize_events_table` enforces them (on a copy or in
place) and returns the resulting node label maps.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

_SOURCES = "source_nodes"
_TARGETS = "target_nodes"
_STARTS = "starting_times"
_ENDINGS = "ending_times"


class SanitizedTable(NamedTuple):
    """Result of :func:`sanitize_events_table`.

    Attributes
    ----------
    events_table : :class:`pandas.DataFrame`
        The sanitized event table (the input object itself if
        ``inplace=True`` was used).
    label_to_node_dict : dict
        Mapping from original node labels to contiguous ``0..N-1`` ids.
        Identity mapping if the input was already contiguous.
    node_to_label_dict : dict
        Inverse of ``label_to_node_dict``.
    """
    events_table: pd.DataFrame
    label_to_node_dict: dict
    node_to_label_dict: dict


def _is_contiguous(src, tgt) -> bool:
    """Check whether node ids are contiguous integers ``0..N-1``.

    Parameters
    ----------
    src, tgt : array-like
        Source and target node columns.

    Returns
    -------
    bool
    """
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    if src.size == 0 and tgt.size == 0:
        return True
    vals = np.unique(np.concatenate([src, tgt]))
    num_nodes = len(vals)
    return bool(vals.dtype.kind in "iu"
                and vals.min() == 0
                and vals.max() == num_nodes - 1)


def _build_label_maps(source_iter, target_iter) -> tuple[dict, dict]:
    """Build label<->node-id dicts from two iterables of node labels.

    Returns
    -------
    (label_to_node_dict, node_to_label_dict) : tuple of dict
        ``label_to_node_dict`` maps original labels to contiguous
        ``0..N-1`` ids (labels sorted ascending); ``node_to_label_dict``
        is its inverse.
    """
    all_nodes = set()
    all_nodes.update(source_iter)
    all_nodes.update(target_iter)
    label_to_node_dict = {m: n for n, m in enumerate(sorted(all_nodes))}
    node_to_label_dict = {n: m for m, n in label_to_node_dict.items()}
    return label_to_node_dict, node_to_label_dict


def _validate_label_map(label_to_node_dict: dict) -> None:
    """Raise ValueError if the provided label map is not a bijection."""
    values = list(label_to_node_dict.values())
    if len(set(values)) != len(values):
        raise ValueError(
            "label_to_node_dict must have unique values for each label."
        )


def needs_sanitization(events_table: pd.DataFrame) -> bool:
    """Check whether an events table violates the internal invariants.

    The invariants required by :class:`~tempnet.ContTempNetwork` (and
    guaranteed by :func:`sanitize_events_table`) are:

    1. node ids are contiguous integers ``0..N-1``,
    2. events are sorted by ``(starting_times, ending_times)``,
    3. the index is a zero-based :class:`pandas.RangeIndex`.

    Parameters
    ----------
    events_table : :class:`pandas.DataFrame`
        Table with at least the columns ``source_nodes``,
        ``target_nodes`` and ``starting_times``. ``ending_times`` is
        used for the sort check when present.

    Returns
    -------
    bool
        ``True`` if :func:`sanitize_events_table` would change the
        table, ``False`` if it already satisfies all invariants.
    """
    # 1. contiguous node ids
    if not _is_contiguous(events_table[_SOURCES], events_table[_TARGETS]):
        return True

    # 2. chronological sort
    sort_cols = [_STARTS]
    if _ENDINGS in events_table.columns:
        sort_cols.append(_ENDINGS)
    if len(events_table) > 1:
        sorted_idx = events_table[sort_cols].apply(tuple, axis=1)
        if not sorted_idx.is_monotonic_increasing:
            return True

    # 3. zero-based RangeIndex (a named index also breaks the internal
    #    reset_index()-based time-grid computation)
    index = events_table.index
    if index.name is not None:
        return True
    if list(index) != list(range(len(events_table))):
        return True

    return False


def sanitize_events_table(
    events_table: pd.DataFrame,
    *,
    label_to_node_dict: dict | None = None,
    inplace: bool = False,
) -> SanitizedTable:
    """Normalize an events table to the internal representation.

    Enforces the three invariants documented in
    :func:`needs_sanitization`: contiguous ``0..N-1`` node ids,
    chronological event order, and a zero-based ``RangeIndex``.

    Parameters
    ----------
    events_table : :class:`pandas.DataFrame`
        Table with at least the columns ``source_nodes``,
        ``target_nodes`` and ``starting_times``.
    label_to_node_dict : dict, optional
        User-provided mapping from original node labels to node ids.
        Must be a bijection and must map the used labels onto
        contiguous ``0..N-1`` ids; otherwise a ``ValueError`` is
        raised. If ``None`` (default), a mapping is built
        automatically when the node ids are not already contiguous.
    inplace : bool, default=False
        If ``True``, modify ``events_table`` in place (no copy). If
        ``False``, work on a copy and leave the input untouched.

    Returns
    -------
    :class:`SanitizedTable`
        Named tuple ``(events_table, label_to_node_dict,
        node_to_label_dict)``.

    Raises
    ------
    ValueError
        If ``label_to_node_dict`` is not a bijection or does not map
        the node labels onto contiguous ``0..N-1`` ids.
    """
    if not inplace:
        events_table = events_table.copy()

    if label_to_node_dict is not None:
        _validate_label_map(label_to_node_dict)
        node_to_label_dict = {v: k for k, v in label_to_node_dict.items()}

        events_table[_SOURCES] = events_table[_SOURCES].map(
            label_to_node_dict)
        events_table[_TARGETS] = events_table[_TARGETS].map(
            label_to_node_dict)

        if not _is_contiguous(events_table[_SOURCES],
                              events_table[_TARGETS]):
            raise ValueError(
                "Nodes not labeled 0..num_nodes-1 after relabeling."
            )
    elif not _is_contiguous(events_table[_SOURCES],
                            events_table[_TARGETS]):
        label_to_node_dict, node_to_label_dict = _build_label_maps(
            events_table[_SOURCES], events_table[_TARGETS],
        )
        events_table[_SOURCES] = events_table[_SOURCES].map(
            label_to_node_dict)
        events_table[_TARGETS] = events_table[_TARGETS].map(
            label_to_node_dict)
    else:
        num_nodes = pd.unique(
            events_table[[_SOURCES, _TARGETS]].values.ravel("K")
        ).size
        label_to_node_dict = {i: i for i in range(num_nodes)}
        node_to_label_dict = {i: i for i in range(num_nodes)}

    sort_cols = [_STARTS]
    if _ENDINGS in events_table.columns:
        sort_cols.append(_ENDINGS)
    events_table.sort_values(by=sort_cols, inplace=True)
    events_table.reset_index(drop=True, inplace=True)
    events_table.index.name = None

    return SanitizedTable(
        events_table=events_table,
        label_to_node_dict=label_to_node_dict,
        node_to_label_dict=node_to_label_dict,
    )
