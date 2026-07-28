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
Toy Temporal Network
====================

This example introduces the core temporal-network workflow in ``tempnet``.
A **temporal network** is a graph whose edges are active only during specific
time intervals: two nodes are connected for a finite period, then disconnect.
Each connection is a tuple ``(u, v, t_start, t_end)``.

We build a small toy network, aggregate it into a static graph, compute the
sequence of random-walk Laplacians, and finally simulate a continuous-time
random walk by exponentiating those Laplacians.
"""
# %%
from matplotlib import pyplot as plt
import seaborn as sns
import tempnet as tn
import networkx as nx

# %%
# Building the temporal network
# -----------------------------
# Consider a small toy network with three edges:
#
# ===== ====== =================
# Edge  Nodes  Active interval
# ===== ====== =================
# 1     0, 1   [0, 1.5]
# 2     1, 2   [1.5, 2.5]
# 3     1, 2   [3.5, 4]
# ===== ====== =================
#
# To construct the temporal network, define four parallel lists -- one each for
# source nodes, target nodes, start times, and end times -- then pass them to
# the constructor. Each index across the four lists corresponds to a single edge.

source_nodes = [0, 1, 1]
target_nodes = [1, 2, 2]
starting_times = [0, 1.5, 3.5]
ending_times = [1.5, 2.5, 4]

tnet = tn.ContTempNetwork(
    source_nodes=source_nodes,
    target_nodes=target_nodes,
    starting_times=starting_times,
    ending_times=ending_times,
)

# %%
# We can print the object to see the number of nodes and events, or access
# them through properties.

print(tnet)
print("num_nodes, num_events:", tnet.num_nodes, tnet.num_events)

# %%
# The full cleaned dataframe is available in one go, including a ``durations``
# column derived from the start and end times.

print(tnet.events_table)

# %%
# Aggregating into a static network
# ---------------------------------
# We can collapse the time dimension entirely, aggregating the temporal network
# into a static graph. This is visualized as a heatmap, where each cell's color
# represents the total weight of edge activations between a pair of nodes.

A = tnet.compute_static_adjacency_matrix().toarray()

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200)
sns.heatmap(A, ax=ax, annot=True, cbar_kws={"label": "Weight"})
ax.set_xlabel("Nodes")
ax.set_ylabel("Nodes")
ax.set_title("Aggregated Network Adjacency Matrix")
plt.show()


# %%
# We then transform it into a NetworkX object to visualise and run other
# algorithms on it.

static = nx.from_numpy_array(A)

pos = nx.circular_layout(static)

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200)
nx.draw(static, pos, with_labels=True, node_color="lightblue", node_size=500)
edge_labels = nx.get_edge_attributes(static, "weight")
nx.draw_networkx_edge_labels(static, pos, edge_labels=edge_labels)
plt.title("Aggregated Static Network")
plt.show()

# %%
# Aggregating into a static network over a period
# -----------------------------------------------
# You can also choose the period to aggregate over by passing
# start_time and end_time to the function.

A_period = tnet.compute_static_adjacency_matrix(start_time=0, end_time=2).toarray()

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200)
sns.heatmap(A_period, ax=ax, annot=True, cbar_kws={"label": "Weight"})
ax.set_xlabel("Nodes")
ax.set_ylabel("Nodes")
ax.set_title("Aggregated Network Adjacency Matrix (t = 0 to 2)")
plt.show()

# %%
# Inspecting nodes and timestamps
# -------------------------------
# Back in the temporal representation, we can access the list of nodes, all
# timestamps, and the start/end of the network (the minimum start time and
# maximum end time).
tnet._compute_time_grid()
print("Node array", tnet.node_array)
print("Timestamps", tnet.times)
print("Start:", tnet.start_time)
print("End:", tnet.end_time)

# %%
# Random-walk Laplacians
# ----------------------
# This package implements the continuous-time random walk on temporal networks,
# which can be used to capture conditional entropy, assortativity, and
# community detection via flow stability.
#
# Given a temporal network with ordered timestamps :math:`t_0, t_1, \dots, t_T`,
# we construct a sequence of graph snapshots. For each consecutive pair
# :math:`(t_i, t_{i+1})`, we extract the subgraph of edges active during that
# interval and compute its **random walk Laplacian**.
#
# For a snapshot with adjacency matrix :math:`A` and degree matrix
# :math:`D = \mathrm{diag}(d_1, \dots, d_n)`, the random walk Laplacian is
#
# .. math::
#
#    L_{\mathrm{rw}} = I - D^{-1}A
#
# where :math:`I` is the identity matrix. If a node :math:`i` has degree
# :math:`d_i = 0` in a given snapshot, :math:`D^{-1}` is undefined; to handle
# this, we make the random walker stay in place by adding a self-loop
# (:math:`A_{ii} = 1`, :math:`d_i = 1`). This yields one Laplacian per interval
# :math:`[t_i, t_{i+1})`.