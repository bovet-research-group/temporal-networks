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

# %%
# Load the data
# -------------
# The dataset ships gzip-compressed as three space-separated columns with no
# header: ``src``, ``dst`` and ``timestamp``. :func:`pandas.read_csv` can read
# it straight from the URL and decompress it.

URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
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
tnet.events_table.head()

# %%
# A quick look at the time span of the data. The timestamps run from
# ``start_time`` to ``end_time`` (in seconds); dividing by the number of
# seconds in a week tells us how many weeks of activity the dataset covers.

n_weeks = tnet.end_time // (7 * 24 * 3600)
print(f"start_time = {tnet.start_time}")
print(f"end_time   = {tnet.end_time}")
print(f"≈ {int(n_weeks)} weeks of activity over {tnet.num_nodes} nodes")
