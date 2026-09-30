# SPDX-FileCopyrightText: 2021 Alexandre Bovet <alexandre.bovet@uzh.ch>
# SPDX-FileCopyrightText: 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>
# SPDX-FileCopyrightText: 2026 Yasaman Asgari <yasaman.asgari@uzh.ch>
# SPDX-FileCopyrightText: 2026 Samuel Koovely <samuel.koovely@uzh.ch>
# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Synthetic temporal network with evolving community structure and conditional entropy computation
====================================================

This example uses :class:`~tempnet.synth_temp_network.SynthTempNetwork` to
generate a synthetic continuous-time temporal network.

The model works as follows:

1. We first define a set of nodes for the network. In this example, 12 agents
   are organized into communities.
2. Each node is assigned an activation rate, ``lambda_activation``.
3. Each node activates on its own (Poissonian) clock, with a waiting time drawn
   from an exponential distribution with mean ``inter_tau=1/lambda_activation``. 
4. Each time a node activates, it chooses ``num_partner_per_activation``
   partner(s) according to the selection strategy (uniform, within-group, or
   block-probability based, here we do an example with blocks).
5. Each resulting interaction gets a duration drawn from another exponential
   with rate ``activ_distro_scale=1/lambda_duration``. When the edge ends, the partner 
   becomes available again for that node's future activations.

In the simulation below, agents are organized into three communities of four.
Within-community contacts are more frequent than cross-community ones (a block
probability structure).

The simulation produces a stream of time-stamped contact events that are then
loaded into a :class:`~tempnet.ContTempNetwork` for analysis.
"""

# %%
# Setup
# -----
# Create 12 :class:`~tempnet.synth_temp_network.Individual` agents split into
# three equal communities (groups 0, 1, 2).  Interaction durations are drawn
# from an exponential distribution with mean ``inter_tau``; inter-activation
# waiting times use mean ``activ_tau``.

import warnings
import matplotlib.pyplot as plt
import numpy as np

from tempnet import ContTempNetwork
from tempnet.synth_temp_network import (Individual,
                                        SynthTempNetwork)

rng = np.random.default_rng(42)

N_GROUPS = 3
N_PER_GROUP = 4
inter_tau = 1.0   # mean contact duration
activ_tau = 5.0   # mean inter-activation time

Individual.all_IDs = []    # reset class-level state between runs
Individual.all_groups = []

individuals = [
    Individual(
        ID=g * N_PER_GROUP + i,
        inter_distro_scale=inter_tau,
        activ_distro_scale=activ_tau,
        group=g,
    )
    for g in range(N_GROUPS)
    for i in range(N_PER_GROUP)
]

def make_step_block_probs(
    deltat1: float,
    deltat2: float,
    m1: float = 1.0,
    p1: float = 1.0,
):
    """Return a time-dependent block-probability function for 3 groups.

    The returned function cycles through three phases where different
    community pairs have elevated cross-community interaction probability.

    Parameters
    ----------
    deltat1 : float
        Duration of each *within-community* (identity-block) phase.
    deltat2 : float
        Duration of each *cross-community exchange* phase.
    m1 : float
        Within-community interaction probability (default 1.0).
    p1 : float
        Cross-community interaction probability for the active pair
        (default 1.0).

    Returns
    -------
    block_mod_func : callable
        A function ``block_mod_func(t)`` that accepts a float *t* and
        returns a 3×3 numpy array of group-level interaction probabilities.
    """
    def block_mod_func(t: float) -> np.ndarray:
        m2 = (1 - m1) / 2
        p2 = (1 - p1)

        ex12 = np.array([[p2, p1, 0],
                         [p1, p2, 0],
                         [0, 0, 0]])
        ex23 = np.array([[0, 0, 0],
                         [0, p2, p1],
                         [0, p1, p2]])
        ex13 = np.array([[p2, 0, p1],
                         [0, 0, 0],
                         [p1, 0, p2]])

        within = np.array([[m1, m2, m2],
                           [m2, m1, m2],
                           [m2, m2, m1]])

        if t >= 0 and t < deltat1:
            return within
        elif t >= deltat1 and t < deltat1 + deltat2:
            return ex12
        elif t >= deltat1 + deltat2 and t < 2 * deltat1 + deltat2:
            return within
        elif t >= 2 * deltat1 + deltat2 and t < 2 * (deltat1 + deltat2):
            return ex23
        elif (t >= 2 * (deltat1 + deltat2)
              and t < 2 * (deltat1 + deltat2) + deltat1):
            return within
        elif (t >= 2 * (deltat1 + deltat2) + deltat1
              and t <= 3 * (deltat1 + deltat2)):
            return ex13
        else:
            warnings.warn(
                "Warning: t must be >=0 and <= 3*(deltat1+deltat2),"
                f" t is {t}"
            )
            return within

    return block_mod_func

# %%
# Block-probability modulation
# ----------------------------
# ``make_step_block_probs`` returns a time-dependent function that cycles
# through phases where different community pairs are highlighted.

m1 = 0.8   # within-community interaction probability
p1 = 0.8   # cross-community interaction probability (for the active pair)

block_prob_mod_func = make_step_block_probs(
    deltat1=40 * activ_tau,
    deltat2=(9 / 2 * m1 - 3 / 2) * 40 * activ_tau / (2 * p1 - 1),
    m1=m1,
    p1=p1,
)
t_end = 3 * (40 * activ_tau + (9 / 2 * m1 - 3 / 2) * 40 * activ_tau / (2 * p1 - 1))

# %%
# Run the simulation
# ------------------

sim = SynthTempNetwork(
    individuals=individuals,
    t_start=0,
    t_end=t_end,
    next_event_method='block_probs_mod',
    block_prob_mod_func=block_prob_mod_func,
)
sim.run()

print(
    f"Simulation produced {len(sim.indiv_sources)} contact events "
    f"over t ∈ [0, {t_end:.1f}]."
)

# %%
# Build a ContTempNetwork
# -----------------------
# The simulated synthetic network is inherently **directed**: the connection
# from ``u`` to ``v`` is generated independently of the one from ``v`` to
# ``u``. As a result, there can be moments where both the ``u → v`` and
# ``v → u`` edges are active simultaneously.
#
# The ``tempnet`` package, however, works with **undirected** networks. When the
# network is constructed, overlapping reciprocal edges are merged into a single
# undirected edge, and the Laplacians are then built from this undirected
# representation.
#
# Note that the Markov property of the resulting process still holds approximately, 
# as long as edge durations are short compared to the inter-event times.

tnet = ContTempNetwork(
    source_nodes=sim.indiv_sources,
    target_nodes=sim.indiv_targets,
    starting_times=sim.start_times,
    ending_times=sim.end_times,
    merge_overlapping_events=True
)

print(f"Nodes : {tnet.node_array}")
print(f"Events: {len(tnet.events_table)}")

# %%
# Plot 1: Contact timeline
# ------------------------
# Each row is a node; each bar represents a contact coloured by the source
# node's community.

GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # one colour per group
node_to_group = {
    g * N_PER_GROUP + i: g
    for g in range(N_GROUPS)
    for i in range(N_PER_GROUP)
}

fig, ax = plt.subplots(figsize=(10, 5))

et = tnet.events_table
for _, row in et.iterrows():
    src = int(row['source_nodes'])
    tgt = int(row['target_nodes'])
    t0 = row['starting_times']
    t1 = row['ending_times']
    color = GROUP_COLORS[node_to_group[src]]
    ax.barh(tgt, t1 - t0, left=t0, height=0.6, color=color, alpha=0.7)

ax.set_xlabel('Time')
ax.set_ylabel('Node')
ax.set_title('Contact timeline — colour indicates source community')

handles = [
    plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g], label=f'Community {g}')
    for g in range(N_GROUPS)
]
ax.legend(handles=handles, loc='upper right')
plt.tight_layout()
plt.show()

# %%
# Plot 2: Event-duration distribution
# ------------------------------------

durations =tnet.events_table['durations'].values

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(durations, bins=30, edgecolor='white')
ax.set_xlabel('Contact duration')
ax.set_ylabel('Count')
ax.set_title('Distribution of contact durations')
plt.tight_layout()
plt.show()

# %%
# Conditional entropy curve
# -------------------------
# In this example the individuals are arranged into different blocks that
# change over time. In such cases the key question is *when* a change occurs
# (change-point detection). Following Koovely et al. (2026), we detect these
# change points from the conditional entropy of the heat diffusion.
#
# The conditional entropy at scale :math:`\tau` is
#
# .. math::
#   S(\tau) = - \sum_i p_i(0) \sum_j T_{ij}(0, \tau) \log T_{ij}(0, \tau),
#
# using the uniform initial distribution over nodes. The curve tracks how the
# temporal activation of edges opens diffusion pathways through the network:
# when new edges appear, heat spreads to a larger portion of the network, which
# shows up as increases in entropy. Flat portions show intervals where the
# available temporal paths do not substantially expand the set of nodes reached
# by the diffusion, and sharp rises flag change points.
#
# The dashed curve is a component-size upper bound. For each time :math:`\tau` it
# aggregates the static graph from the start of the network up to :math:`\tau` and
# finds its connected components. Heat starting in a component of size
# :math:`|C|` cannot reach more than :math:`|C|` nodes, so that component's
# entropy is bounded by :math:`\log |C|`. Averaged over components,
#
# .. math::
#   \sum_C \frac{|C|}{N} \log |C|,
#
# isolated nodes contribute zero and the largest possible value is
# :math:`\log N`, reached only when all nodes lie in a single cumulative
# component.
#
# The backward-in-time panel repeats the analysis with the diffusion and the
# aggregation window reversed: paths and cumulative components are built from
# the end of the observation window back to the start. Comparing the two panels
# shows temporal asymmetry, a change point in one direction is
# not necessarily a change point in the other.

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharey=True)

scales = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

tnet.compute_laplacian_matrices(dynamics='heat', save_adjacencies=False)
times = tnet.times

for lamda in scales:
    tnet.compute_inter_transition_matrices(lamda=lamda, method='dense_expm')

# Forward in time
for lamda in scales:
    tnet.compute_transition_matrices(lamda=lamda, save_intermediate=True, reverse_time=False)
    tnet.compute_conditional_entropy_curve(lamda=lamda, time_downsampling_factor=0.25)
    S = tnet.S[lamda]
    axes[0].plot(times[S[:, 0].astype(int)], S[:, 1], label=rf"$\lambda$={lamda}")

tnet.compute_entropy_upper_bound_curve(time_downsampling_factor=0.25)
bound = tnet.S_upper_bound
axes[0].plot(times[bound[:, 0].astype(int)], bound[:, 1],
             label="Upper bound", linestyle='--', color='black')

# Reset cached state before the backward pass, otherwise the backward panel silently reuses the forward bound.
for attr in ('T', 'S', 'direction', 'S_upper_bound'):
    if hasattr(tnet, attr):
        delattr(tnet, attr)

# Backward in time
for lamda in scales:
    tnet.compute_transition_matrices(lamda=lamda, save_intermediate=True, reverse_time=True)
    tnet.compute_conditional_entropy_curve(lamda=lamda, time_downsampling_factor=0.25)
    S = tnet.S[lamda]
    axes[1].plot(times[S[:, 0].astype(int)], S[:, 1], label=rf"$\lambda$={lamda}")

tnet.compute_entropy_upper_bound_curve(time_downsampling_factor=0.25)
bound = tnet.S_upper_bound
axes[1].plot(times[bound[:, 0].astype(int)], bound[:, 1],
             label="Upper bound", linestyle='--', color='black')

axes[0].set_title('Forward in time')
axes[1].set_title('Backward in time')
axes[1].set_xlabel("Time")
axes[0].set_ylabel("Conditional Entropy (nats)")
axes[1].set_ylabel("Conditional Entropy (nats)")

axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
fig.tight_layout()
plt.show()