"""
==================================================================
EU emails an an instantaneous temporal network
==================================================================

This example walks through a minimal analysis pipeline with
:mod:`tempnet` on a real-world dataset: the
`email-Eu-core-temporal <https://snap.stanford.edu/data/email-Eu-core-temporal.html>`_
network from the Stanford SNAP collection.

The email data corresponds to a large European research institution and
each row of the dataset is a single, *instantaneous* email: a sender, a
recipient, and the time (in seconds, from the start of the observation) at
which the message was sent. Because the events have no duration we model the
data with :class:`~tempnet.temporal_network.ContTempInstNetwork`, the
instantaneous-event variant of the continuous-time temporal network.

The pipeline has three steps:
1. load the events into a temporal network and explore basic properties
2. build the Laplacian matrices for a one-week window, and
3. using different rates to understand the dynamics of the emailing. 
"""

# %%
# Setup
# -----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tempnet.temporal_network import ContTempInstNetwork
from functools import reduce

import seaborn as sns
from matplotlib.colors import LogNorm

# %%
# Load the data
# -------------
# The dataset ships gzip-compressed as three space-separated columns with no
# header: ``src``, ``dst`` and ``timestamp``. :func:`pandas.read_csv` can read
# it straight from the URL and decompress it.

URL = "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept3.txt.gz"
df = pd.read_csv(URL, compression="gzip", sep=" ", names=["src", "dst", "timestamp"])

df.head()

# %%
# Build the temporal network
# ---------------------------
# We hand the three columns to :class:`~tempnet.temporal_network.ContTempInstNetwork`.
# ``relabel_nodes=True`` remaps the raw node ids to a compact ``0..N-1`` range.

tnet = ContTempInstNetwork(
    source_nodes=df["src"],
    target_nodes=df["dst"],
    starting_times=df["timestamp"],
    relabel_nodes=True,
)
print(tnet)

# %%
# The events are stored in a tidy table we can inspect directly.
print(tnet.events_table.head())

# %%
# A quick look at the time span of the data. The timestamps run from
# ``start_time`` to ``end_time`` (in seconds); dividing by the number of
# seconds in a week tells us how many weeks of activity the dataset covers.

n_weeks = tnet.end_time // (7 * 24 * 3600)
print(f"start_time = {tnet.start_time}")
print(f"end_time   = {tnet.end_time}")
print(f"≈ {int(n_weeks)} weeks of activity over {tnet.num_nodes} nodes")

# %%
# Restrict to the first week
# --------------------------
# The full dataset spans well over a year, which is far more events than we
# need to illustrate the method. We compute the per-event Laplacian matrices
# for the **first week only** by passing ``t_stop`` (in seconds).

ONE_WEEK = 7 * 24 * 3600
tnet.compute_laplacian_matrices(t_start=None, t_stop=ONE_WEEK)

# %%
scales=[1, 10, 60,24*3600]
for scale in scales: 
    tnet.compute_inter_transition_matrices(lamda=1/scale, method='parallel_expm',n_jobs=1,nproc= 4, normalize_rows= True)

# %%
from tqdm import tqdm
forward_transition_matrices = [
    reduce(lambda a, b: a @ b, tqdm(tnet.inter_T[1/s], leave=False))
    for s in tqdm(scales)
]

# %%
# Visualise the forward transition matrices for each time scale.
norm = LogNorm(vmin=1e-6, vmax=1)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16), dpi=500)
ax=ax.flat
for i, (lamda, matrix) in enumerate(zip(scales, forward_transition_matrices)):
    sns.heatmap(matrix.toarray(), ax=ax[i], square=True, cbar=False,
                norm=norm)
    ax[i].set_title(rf'$\lambda$={lamda}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])

#fig.colorbar(ax[0].collections[0], ax=ax, label='Transition probability',
    #         fraction=0.046, pad=0.04)
plt.show()

