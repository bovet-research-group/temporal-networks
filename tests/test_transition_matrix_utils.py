import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csr
from stochmat import SparseStochMat

from tempnet.utils import (
    _prepare_inter_transition_matrix,
    _threshold_and_row_normalize,
)


def test_threshold_and_row_normalize_csr_in_place():
    """CSR matrices are thresholded and row-normalized in place."""
    matrix = csr_matrix([
        [1.0, 1e-4],
        [0.0, 2.0],
    ])

    result = _threshold_and_row_normalize(matrix, tol=1e-2)

    assert result is matrix
    np.testing.assert_allclose(result.toarray(), np.eye(2))


def test_threshold_and_row_normalize_csc_warns_and_returns_copy():
    """CSC inputs warn and return a row-normalized copy."""
    matrix = csc_matrix([
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    original = matrix.copy()

    with pytest.warns(UserWarning, match="CSC transition matrices"):
        result = _threshold_and_row_normalize(matrix, tol=None)

    assert isinstance(result, csc_matrix)
    assert result is not matrix
    np.testing.assert_allclose(result.toarray().sum(axis=1), np.ones(2))
    np.testing.assert_allclose(matrix.toarray(), original.toarray())


def test_prepare_inter_transition_matrix_copies_csr_before_cleaning():
    """Preparing a CSR inter-event matrix must not mutate the source."""
    inter_matrix = csr_matrix([
        [1.0, 1e-4],
        [0.0, 2.0],
    ])
    before = inter_matrix.copy()

    result = _prepare_inter_transition_matrix(
        inter_matrix,
        force_csr=False,
        tol=1e-2,
    )

    assert result is not inter_matrix
    np.testing.assert_allclose(inter_matrix.toarray(), before.toarray())
    np.testing.assert_allclose(result.toarray(), np.eye(2))


def test_prepare_inter_transition_matrix_force_csr_returns_csr_copy():
    """force_csr=True converts CSC input to a normalized CSR copy."""
    inter_matrix = csc_matrix([
        [1.0, 0.0],
        [0.0, 2.0],
    ])

    result = _prepare_inter_transition_matrix(
        inter_matrix,
        force_csr=True,
        tol=None,
    )

    assert isspmatrix_csr(result)
    assert result is not inter_matrix
    np.testing.assert_allclose(result.toarray().sum(axis=1), np.ones(2))


def test_prepare_inter_transition_matrix_sparse_stoch_requires_force_csr():
    """SparseStochMat inputs require force_csr=True for products."""
    inter_matrix = SparseStochMat.create_diag(size=2)

    with pytest.raises(ValueError, match="force_csr"):
        _prepare_inter_transition_matrix(
            inter_matrix,
            force_csr=False,
            tol=None,
        )


def test_prepare_inter_transition_matrix_sparse_stoch_force_csr():
    """SparseStochMat inputs are converted to CSR with force_csr=True."""
    inter_matrix = SparseStochMat.create_diag(size=2)

    result = _prepare_inter_transition_matrix(
        inter_matrix,
        force_csr=True,
        tol=None,
    )

    assert isspmatrix_csr(result)
    np.testing.assert_allclose(result.toarray(), np.eye(2))
