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

"""Regression tests for gallery example data downloads.

``test_plot_02_uses_zenodo_get`` guards against reverting to ``pd.read_csv(URL)``
or ``requests.get``, both of which hang on Zenodo's redirect chain.

The correct pattern — ``zenodo_get.download()`` + ``tempfile.TemporaryDirectory()``
— is tested in both the source-inspection test and the live network test.
"""

import re
import tempfile
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "example"

RECORD_ID = "4725155"
FILE_NAME = "mice_contact_sequence.csv.gz"


def test_plot_02_uses_zenodo_get():
    """plot_02 must download via zenodo_get.download() (handles redirects + timeout).

    ``pd.read_csv(URL)`` and ``requests.get(URL)`` both hang because they do not
    follow Zenodo's redirect chain correctly and/or have no built-in timeout.
    The correct fix is to use ``from zenodo_get import download`` and call
    ``download(record_or_doi=..., output_dir=..., file_glob=...)``.
    """
    source = (EXAMPLES_DIR / "plot_02_real_network.py").read_text()

    assert "from zenodo_get import download" in source, (
        "plot_02 must import download from zenodo_get"
    )
    assert "download(" in source, (
        "plot_02 must call download() to fetch the Zenodo dataset"
    )
    assert "requests.get(" not in source, (
        "plot_02 must not call requests.get() — use zenodo_get.download() instead"
    )
    assert not re.search(r"pd\.read_csv\(['\"]https?://", source), (
        "pd.read_csv must not be called with a raw URL — "
        "use zenodo_get.download() + Path(tmpdir) / FILE_NAME instead"
    )


@pytest.mark.network
def test_zenodo_mice_dataset_accessible():
    """The Zenodo record must deliver a valid gzip CSV via zenodo_get.

    Uses the same download pattern as the fixed plot_02 example so that
    redirect handling, built-in timeout, and CSV parsing are all exercised.
    """
    import pandas as pd
    from zenodo_get import download

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading {FILE_NAME} from Zenodo record {RECORD_ID}...")
        download(
            record_or_doi=RECORD_ID,
            output_dir=tmpdir,
            file_glob=FILE_NAME,
        )
        print("Loading into pandas...")
        df = pd.read_csv(Path(tmpdir) / FILE_NAME, compression="gzip")

    print(f"Success! Dataset shape: {df.shape}")
    print(df.head())

    assert len(df) > 0, "Dataset must not be empty"
    assert {
        "source_nodes", "target_nodes", "starting_times", "ending_times"
    }.issubset(df.columns), (
        f"Missing expected columns. Got: {list(df.columns)}"
    )
