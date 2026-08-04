import os

import numpy as np
import pytest
from scipy.sparse import block_diag, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import expm

from tempnet.faster_expm import (
    _stack_sparse_cols,
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


def test_compute_parallel_expm_accepts_non_csc_input():
    A = -_two_component_laplacian().tocsr()
    expected = expm(A).toarray()

    actual = compute_parallel_expm(
        A,
        nproc=NPROC,
        normalize_rows=False,
    ).toarray()

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_stack_sparse_cols_reconstructs_square_csc_matrix():
    cols = [
        csc_matrix([[1.0], [0.0], [2.0]]),
        csc_matrix([[0.0], [3.0], [0.0]]),
        csc_matrix([[4.0], [0.0], [5.0]]),
    ]

    actual = _stack_sparse_cols(cols)

    np.testing.assert_allclose(
        actual.toarray(),
        np.array([
            [1.0, 0.0, 4.0],
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 5.0],
        ]),
    )


def test_stack_sparse_cols_rejects_non_square_input():
    cols = [
        csc_matrix([[1.0], [0.0], [2.0]]),
        csc_matrix([[0.0], [3.0], [0.0]]),
    ]

    with pytest.raises(ValueError, match="N columns"):
        _stack_sparse_cols(cols)


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


def test_compute_subspace_expm_normalizes_rows_after_thresholding():
    A = -_path_graph_laplacian(size=50)
    thresh_ratio = 100.0

    raw = compute_subspace_expm(
        A.copy(),
        thresh_ratio=thresh_ratio,
        normalize_rows=False,
    ).toarray()
    assert not np.allclose(raw.sum(axis=1), np.ones(A.shape[0]))

    normalized = compute_subspace_expm(
        A.copy(),
        thresh_ratio=thresh_ratio,
        normalize_rows=True,
    ).toarray()

    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(A.shape[0]))


def test_compute_subspace_expm_parallel_normalizes_rows_after_thresholding():
    A = -_path_graph_laplacian(size=50)
    thresh_ratio = 100.0

    raw = compute_subspace_expm_parallel(
        A.copy(),
        nproc=NPROC,
        thresh_ratio=thresh_ratio,
        normalize_rows=False,
    ).toarray()
    assert not np.allclose(raw.sum(axis=1), np.ones(A.shape[0]))

    normalized = compute_subspace_expm_parallel(
        A.copy(),
        nproc=NPROC,
        thresh_ratio=thresh_ratio,
        normalize_rows=True,
    ).toarray()

    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(A.shape[0]))


def test_compute_subspace_expm_parallel_uses_large_component_branch(
    monkeypatch,
):
    L = _path_graph_laplacian(size=1001)
    calls = []

    def fake_compute_parallel_expm(A, *, nproc, thresh_ratio, normalize_rows):
        calls.append((A.shape, nproc, thresh_ratio, normalize_rows))
        return csc_matrix(np.eye(A.shape[0]))

    monkeypatch.setattr(
        "tempnet.faster_expm.compute_parallel_expm",
        fake_compute_parallel_expm,
    )

    actual = compute_subspace_expm_parallel(
        -L,
        nproc=NPROC,
        normalize_rows=False,
    )

    assert calls == [((1001, 1001), NPROC, None, False)]
    np.testing.assert_allclose(actual.toarray(), np.eye(1001))


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


def test_sparse_lapl_expm_sparse_small_matrix_matches_dense():
    L = _two_component_laplacian()
    expected = expm(-L).toarray()

    actual = to_dense(sparse_lapl_expm(L.copy(), fact=1.0,
                                      dense_expm=False))

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_sparse_lapl_expm_sparse_large_single_component_uses_sparse_expm(
    monkeypatch,
):
    L = _path_graph_laplacian(size=1000)
    calls = []

    def fake_expm(A):
        calls.append(A.shape)
        return csr_matrix(np.eye(A.shape[0]))

    monkeypatch.setattr("tempnet.faster_expm.expm", fake_expm)

    actual = to_dense(sparse_lapl_expm(L.copy(), fact=1.0,
                                      dense_expm=False))

    assert calls == [(1000, 1000)]
    np.testing.assert_allclose(actual, np.eye(1000))


def test_sparse_lapl_expm_sparse_large_multi_component_uses_parallel_subspace(
    monkeypatch,
):
    L_component = _path_graph_laplacian(size=500)
    L = block_diag((L_component, L_component), format="csc")
    calls = []

    def fake_parallel(A, *, n_comp, comp_labels, nproc, thresh_ratio,
                      normalize_rows):
        calls.append((A.shape, n_comp, nproc, thresh_ratio, normalize_rows))
        return csr_matrix(np.eye(A.shape[0]))

    monkeypatch.setattr(
        "tempnet.faster_expm.compute_subspace_expm_parallel",
        fake_parallel,
    )

    actual = to_dense(
        sparse_lapl_expm(
            L.copy(),
            fact=1.0,
            dense_expm=False,
            nproc=NPROC,
            normalize_rows=False,
        )
    )

    assert calls == [((1000, 1000), 2, NPROC, None, False)]
    np.testing.assert_allclose(actual, np.eye(1000))


def test_sparse_lapl_expm_zero_laplacian_returns_identity():
    L = csr_matrix((4, 4))

    actual = to_dense(sparse_lapl_expm(L, fact=1.0))

    np.testing.assert_allclose(actual, np.eye(4))
