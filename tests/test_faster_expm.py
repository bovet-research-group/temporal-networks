import os

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import expm

from tempnet.faster_expm import (
    compute_parallel_expm,
    compute_subspace_expm,
    compute_subspace_expm_parallel,
    sparse_lapl_expm,
)
from tempnet.utils import to_dense


NPROC = max(2, os.cpu_count() or 1)


def _two_component_laplacian():
    """Small block-diagonal Laplacian with two disconnected components."""
    return csc_matrix([
        [1.0, -1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, -1.0, 1.0],
    ])


def _path_graph_laplacian(size):
    nodes = np.arange(size - 1)
    rows = np.concatenate([nodes, nodes + 1])
    cols = np.concatenate([nodes + 1, nodes])
    data = np.ones(rows.shape[0], dtype=np.float64)

    adjacency = csc_matrix((data, (rows, cols)), shape=(size, size))
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return diags(degrees, format="csc") - adjacency


def test_compute_subspace_expm_matches_dense_expm_on_disconnected_graph():
    A = -_two_component_laplacian()
    expected = expm(A).toarray()

    actual = compute_subspace_expm(
        A.copy(),
        normalize_rows=False,
    ).toarray()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_compute_subspace_expm_parallel_matches_dense_expm():
    A = -_two_component_laplacian()
    expected = expm(A).toarray()

    actual = compute_subspace_expm_parallel(
        A.copy(),
        nproc=NPROC,
        normalize_rows=False,
    ).toarray()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_compute_parallel_expm_matches_dense_expm():
    A = -_two_component_laplacian()
    expected = expm(A).toarray()

    actual = compute_parallel_expm(
        A.copy(),
        nproc=NPROC,
        normalize_rows=False,
    ).toarray()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_compute_parallel_expm_normalizes_rows_after_thresholding():
    """Thresholding trims small entries, so raw row sums drop below 1;
    normalize_rows must restore them to exactly 1.
    """
    A = -_path_graph_laplacian(size=50)
    thresh_ratio = 100.0  # trims entries < max(col)/100

    raw = compute_parallel_expm(
        A.copy(),
        nproc=NPROC,
        thresh_ratio=thresh_ratio,
        normalize_rows=False,
    ).toarray()
    # sanity: without normalization, thresholding breaks row-stochasticity
    assert not np.allclose(raw.sum(axis=1), np.ones(A.shape[0]))

    normalized = compute_parallel_expm(
        A.copy(),
        nproc=NPROC,
        thresh_ratio=thresh_ratio,
        normalize_rows=True,
    ).toarray()

    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(A.shape[0]))


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Large multiprocessing branch test is intended for local runs.",
)
def test_compute_subspace_expm_parallel_computes_large_component_branch():
    L = _path_graph_laplacian(size=1001)
    expected = expm((-L).toarray())

    actual = compute_subspace_expm_parallel(
        -L,
        nproc=NPROC,
        normalize_rows=False,
    )

    assert actual.shape == L.shape
    assert actual.getnnz() > L.shape[0]
    np.testing.assert_allclose(actual.toarray(), expected,
                               rtol=1e-8, atol=1e-10)


def test_sparse_lapl_expm_matches_dense_laplacian_expm():
    L = _two_component_laplacian()
    fact = 0.7
    expected = expm(-fact * L).toarray()

    actual = to_dense(
        sparse_lapl_expm(
            L.copy(),
            fact=fact,
            dense_expm=True,
            normalize_rows=False,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_sparse_lapl_expm_zero_laplacian_returns_identity():
    L = csr_matrix((4, 4))

    actual = to_dense(sparse_lapl_expm(L, fact=1.0))

    np.testing.assert_allclose(actual, np.eye(4))
