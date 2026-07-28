"""Regression tests for gallery example data downloads.

``test_plot_02_uses_zenodo_get`` guards against reverting to ``pd.read_csv(URL)``
or ``requests.get``, both of which hang on Zenodo's redirect chain.

The correct pattern — ``zenodo_get.download()`` + ``tempfile.TemporaryDirectory()``
— is tested in both the source-inspection test and the live network test.
"""

import tempfile
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

RECORD_ID = "4725155"
FILE_NAME = "mice_contact_sequence.csv.gz"


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
