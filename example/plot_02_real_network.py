"""
Mouse contact network (empirical data)
=======================================

This example loads the mouse proximity-contact dataset published by
`König (2021) <https://doi.org/10.5281/zenodo.4725155>`_ and
analyses it as a continuous-time temporal network using
:class:`~tempnet.ContTempNetwork`.

The dataset records pairwise contact events (start time, end time, source
mouse, target mouse) collected over several days.  We restrict the analysis
to the **first hour** (3 600 s) to keep the example fast.

Two plots are produced:

1. **Contact timeline** — each contact is drawn as a horizontal bar; rows
   correspond to individual mice.
2. **Event-duration distribution** — histogram of contact durations.

.. note::

   Downloading the dataset requires `zenodo-get
   <https://pypi.org/project/zenodo-get/>`_, which is **not** a core
   ``tempnet`` dependency.  Install it separately before running this
   example::

       pip install zenodo-get
"""

# %%
# Load the dataset
# ----------------
# The CSV is fetched from Zenodo using :func:`zenodo_get.download`, which
# correctly handles Zenodo's redirect chain and has a built-in timeout.
# The file is written to a temporary directory and loaded into memory;
# the temporary directory is deleted automatically once the data is in RAM.
#
# .. note::
#    ``zenodo-get`` must be installed separately: ``pip install zenodo-get``

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from zenodo_get import download

from tempnet import ContTempNetwork

RECORD_ID = "4725155"
FILE_NAME = "mice_contact_sequence.csv.gz"

with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Downloading {FILE_NAME} from Zenodo record {RECORD_ID}...")
    download(
        record_or_doi=RECORD_ID,
        output_dir=tmpdir,
        file_glob=FILE_NAME,
    )
    print("Loading into pandas...")
    raw_df = pd.read_csv(Path(tmpdir) / FILE_NAME, compression="gzip")
# tmpdir and its contents are deleted here

print(f"Full dataset: {len(raw_df)} events, "
      f"{raw_df['source_nodes'].nunique()} mice")

CUT_AFTER = 3600  # first hour only
events_table = raw_df[raw_df['ending_times'] < CUT_AFTER].copy()
print(f"After {CUT_AFTER} s cut: {len(events_table)} events")

# %%
# Build a ContTempNetwork
# -----------------------

network = ContTempNetwork(
    events_table=events_table,
    merge_overlapping_events=False,
)

nodes = network.nodes
print(f"Nodes : {nodes}")
print(f"Events: {len(network.events_table)}")
print(
    f"Time span: [{network.events_table[ContTempNetwork._STARTS].min():.1f}, "
    f"{network.events_table[ContTempNetwork._ENDINGS].max():.1f}] s"
)

# %%
# Plot 1: Contact timeline
# ------------------------

fig, ax = plt.subplots(figsize=(10, 5))

et = network.events_table
for _, row in et.iterrows():
    tgt = row[ContTempNetwork._TARGETS]
    t0 = row[ContTempNetwork._STARTS]
    t1 = row[ContTempNetwork._ENDINGS]
    ax.barh(tgt, t1 - t0, left=t0, height=0.6, color='steelblue', alpha=0.6)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Mouse ID')
ax.set_title(f'Mouse contact timeline — first {CUT_AFTER // 60} min')
plt.tight_layout()
plt.show()

# %%
# Plot 2: Event-duration distribution
# ------------------------------------

durations = (
    et[ContTempNetwork._ENDINGS] - et[ContTempNetwork._STARTS]
).values

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(durations, bins=30, edgecolor='white', color='steelblue')
ax.set_xlabel('Contact duration (s)')
ax.set_ylabel('Count')
ax.set_title('Distribution of contact durations')
plt.tight_layout()
plt.show()
