"""
#
# Temporal networks `tempnet`
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@uzh.ch>
# Copyright (C) 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>, 
#                    Yasaman Asgari <yasaman.asgari@uzh.ch>, 
#                    Samuel Koovely <samuel.koovely@uzh.ch>, 
#                    Jonas I. Liechti <j-i-l@t4d.ch>
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


"""
Synthetic temporal network with community structure
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
                                        SynthTempNetwork,
                                        make_step_block_probs)

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

durations = et['durations'].values

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(durations, bins=30, edgecolor='white')
ax.set_xlabel('Contact duration')
ax.set_ylabel('Count')
ax.set_title('Distribution of contact durations')
plt.tight_layout()
plt.show()
