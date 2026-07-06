"""Tests for tolerance-filtered sparse matrix exponentials."""

import numpy as np
from scipy.linalg import expm as dense_expm
from scipy.sparse import csr_matrix, issparse

from tempnet.expm_with_tol import flitoutA2, getMN, getfi, mfp_exp


def test_mfp_exp_matches_dense_expm_for_symmetric_laplacian() -> None:
    """mfp_exp should approximate dense expm for a symmetric Laplacian."""
    H = csr_matrix(
        -0.5
        * np.array(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 1.0],
            ]
        )
    )

    T = mfp_exp(H, err=1e-10, non_norm=0)

    assert isinstance(T, csr_matrix)
    np.testing.assert_allclose(T.toarray(), dense_expm(H.toarray()), atol=1e-8)


def test_mfp_exp_matches_dense_expm_for_non_normal_matrix() -> None:
    """mfp_exp should approximate dense expm for a non-normal matrix."""
    H = csr_matrix(
        np.array(
            [
                [-1.0, 1.0, 0.0],
                [0.0, -2.0, 2.0],
                [1.0, 0.0, -1.0],
            ]
        )
    )

    T = mfp_exp(H, err=1e-10, non_norm=1)

    assert isinstance(T, csr_matrix)
    np.testing.assert_allclose(T.toarray(), dense_expm(H.toarray()), atol=1e-8)


def test_flitoutA2_returns_csr_matrix() -> None:
    """flitoutA2 should return a CSR matrix and updated scale."""
    A = csr_matrix(
        np.array(
            [
                [1.0, 1e-12, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 1e-12, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 1e-12, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 1e-12, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 1e-12],
                [1e-12, 0.0, 0.0, 0.0, 0.0, 6.0],
            ]
        )
    )

    A_filtered, scale = flitoutA2(A, eg=1e-6, m=1.0)

    assert isinstance(A_filtered, csr_matrix)
    assert scale >= 1.0
    assert issparse(A_filtered)


def test_getfi_decreases_with_taylor_order() -> None:
    """Taylor truncation error should decrease with the order."""
    assert getfi(8, 0.5) < getfi(4, 0.5)


def test_getMN_returns_positive_integer_parameters() -> None:
    """getMN should return integer Taylor and scaling parameters."""
    M, N = getMN(2.5, 1e-8)

    assert isinstance(M, int)
    assert isinstance(N, int)
    assert M > 0
    assert N > 0
