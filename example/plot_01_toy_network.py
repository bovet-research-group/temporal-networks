"""
Toy Temporal Network
=================================================

This example introduces the core temporal-network workflow in ``tempnet``.
A **temporal network** is a graph whose edges are active only during specific
time intervals: two nodes are connected for a finite period, then disconnect.
Each connection is a tuple ``(u, v, t_start, t_end)``.

We build a small toy network, aggregate it into a static graph, compute
random-walk transition matrices, and finally switch to heat diffusion for the
conditional entropy plot.
"""
# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tempnet as tn
import networkx as nx
from pathlib import Path

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

tnet.compute_laplacian_matrices(dynamics="rw")

# %%
# We can directly access the delta Laplacian matrices for inspection.

fig, ax = plt.subplots(nrows=1, ncols=len(tnet.laplacians), figsize=(16, 4))
for i, L in enumerate(tnet.laplacians):
    sns.heatmap(
        L.toarray(),
        ax=ax[i],
        square=True,
        annot=True,
        cbar=False,
        vmin=-1,
        vmax=1,
        cmap="seismic",
    )
    ax[i].set_title(
        rf"$t_{{\text{{start}}}}$={tnet.times[i]}"
        "\t"
        rf"$t_{{\text{{end}}}}$={tnet.times[i + 1]}"
    )
fig.suptitle("Random-walk delta Laplacians")
plt.show()

# %%
# Transition matrices
# -------------------
# With the random-walk Laplacians computed, we simulate the continuous-time
# random walk by computing the **matrix exponential** of each Laplacian, scaled
# by the duration of the corresponding interval and the walker's transition
# rate. For two consecutive timestamps :math:`t_1` and :math:`t_2`,
#
# .. math::
#   \hat{T}(t_1, t_2; \lambda) = e^{-(t_2 - t_1)\lambda L_{\mathrm{rw}}}
#
# where :math:`\lambda` is the rate of the random walker. The entry
# :math:`\hat{T}_{jk}` gives the probability that a walker starting at node
# :math:`j` at time :math:`t_1` reaches node :math:`k` at time :math:`t_2`.

lamda = 1
tnet.compute_inter_transition_matrices(lamda=lamda, dynamics="rw")

fig, ax = plt.subplots(nrows=1, ncols=len(tnet.inter_T[lamda]), figsize=(16, 4))
for i, matrix in enumerate(tnet.inter_T[lamda]):
    sns.heatmap(
        matrix.toarray(),
        ax=ax[i],
        square=True,
        annot=True,
        cbar=False,
        fmt=".3f",
        vmin=0,
        vmax=1,
    )
    ax[i].set_title(
        rf"$t_{{\text{{start}}}}$={tnet.times[i]}"
        "\t"
        rf"$t_{{\text{{end}}}}$={tnet.times[i + 1]}"
    )
fig.suptitle(r"Random-walk inter transition matrices for $\lambda=1$")
plt.show()

# %%
# Forward transition matrix
# -------------------------
# The **forward transition matrix** is the product of the inter-transition
# matrices:
#
# .. math::
#   T(t_1, t_2; \lambda) = \hat{T}(t_1, t_m; \lambda) \left[ \prod_{k=m}^{n-1} \hat{T}(t_k, t_{k+1}; \lambda) \right] \hat{T}(t_n, t_2; \lambda)
#
# with :math:`m < n`, :math:`t_m \geq t_1` being the time of the first event
# after, or at, :math:`t_1` and :math:`t_n < t_2` the time of the last event
# before :math:`t_2`. To compute the transition matrix corresponding to the
# time-reversed evolution of the network, from :math:`t_2` to :math:`t_1`, we
# perform the matrix product in the reversed order.
#
# The entry :math:`T_{jk}` gives the probability that a walker with rate
# :math:`\lambda`, starting at node :math:`j` at the beginning of the network,
# arrives at node :math:`k` by the end. The rate controls the walker's
# exploration speed:
#
# - **Low rate** (:math:`\lambda \ll 1`): the walker barely moves, so
#   :math:`T` remains close to the identity matrix.
# - **High rate** (:math:`\lambda \gg 1`): the walker mixes rapidly, washing
#   out temporal structure.


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, lamda in enumerate([1e-2, 0.1, 10]):
    tnet.compute_inter_transition_matrices(lamda=lamda, dynamics="rw")
    tnet.compute_transition_matrices(
        lamda=lamda,
        dynamics="rw",
        save_intermediate=False,
        reverse_time=False,
        force_csr=False,
    )
    sns.heatmap(
        tnet.T[lamda].toarray(),
        ax=ax[i],
        square=True,
        annot=True,
        cbar=False,
        fmt=".2f",
        vmin=0,
        vmax=1,
    )
    ax[i].set_title(rf"$\lambda$={lamda}")
fig.suptitle("Forward random-walk transition matrices")
plt.show()

# %%
# Conditional entropy curve
# -------------------------
# We now switch to heat diffusion for the entropy signal:
#
# .. math::
#   S(t) = - \sum_i p_i(0) \sum_j T_{ij}(0, t) \log T_{ij}(0, t)
#
# where we use the uniform initial distribution over nodes. From a network
# science perspective, the entropy curve tracks how the temporal activation of
# edges opens diffusion pathways through the network. When new edges appear,
# heat can spread faster and reach a larger portion of the network, which is
# reflected by increases in entropy production. Flat portions indicate time
# intervals where the currently available temporal paths do not substantially
# expand the set of nodes reached by the diffusion.
#
# Calling ``compute_entropy`` computes the cumulative transition matrices when
# needed, stores the full entropy signal in ``tnet.S[lambda]``, and returns a
# two-column array with transition indices and entropy values.
#
# The dashed curve is a component-size upper bound, not a second diffusion
# process. For each time ``t``, it aggregates the static graph from the start
# of the network up to ``t`` and finds its connected components. If a cumulative
# component has size ``|C|``, heat starting inside it cannot spread to more than
# ``|C|`` nodes, so its entropy contribution is bounded by ``log(|C|)``. The
# plotted bound averages this over components:
#
# .. math::
#   \sum_C \frac{|C|}{N} \log |C|
#
# Isolated nodes contribute zero, and the largest possible value is
# ``log(N)``, reached only when all nodes are in one cumulative component. The
# entropy curve can remain below this upper bound because temporal ordering can
# make paths asymmetric: even when the cumulative graph is connected, not all
# nodes are equally reachable through time-respecting diffusion paths.

entropy_lamda = 1
entropy_dynamics = "heat"
# This recomputes the needed heat-diffusion matrices because the transition
# matrices above were computed for random-walk dynamics.
entropy_curve = tnet.compute_entropy(
    lamda=entropy_lamda,
    dynamics=entropy_dynamics,
)
entropy_indices = entropy_curve[:, 0].astype(int)
entropy_values = np.asarray(tnet.S[entropy_lamda])
entropy_upper_bound = tnet.compute_entropy_upper_bound(return_times=True)

# Each cumulative transition index k corresponds to heat diffusion after the
# interval ending at times[k + 1].
time_values = np.asarray(tnet.times, dtype=float)
entropy_times = time_values[
    tnet._k_start_laplacians + entropy_indices + 1
]

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200)
ax.plot(entropy_times, entropy_values, marker="o", label="Entropy")
ax.plot(
    entropy_upper_bound[:, 0],
    entropy_upper_bound[:, 1],
    color="black",
    linestyle="--",
    label="Component-size upper bound",
)
ax.set_xlabel("Time")
ax.set_ylabel("Conditional entropy")
ax.set_title(rf"Heat diffusion entropy curve for $\lambda={entropy_lamda}$")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(
    Path(__file__).with_name("plot_01_toy_network_entropy_upper_bound.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.show()
