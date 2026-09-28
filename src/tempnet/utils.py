# SPDX-FileCopyrightText: 2021 Alexandre Bovet <alexandre.bovet@uzh.ch>
# SPDX-FileCopyrightText: 2026 Alexandre Bovet <alexandre.bovet@uzh.ch>
# SPDX-FileCopyrightText: 2026 Yasaman Asgari <yasaman.asgari@uzh.ch>
# SPDX-FileCopyrightText: 2026 Samuel Koovely <samuel.koovely@uzh.ch>
# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sparse-matrix utility functions for temporal-network computations."""

import warnings

import numpy as np
from numpy.typing import ArrayLike
from stochmat import inplace_csr_row_normalize, SparseStochMat

from scipy.sparse.linalg import eigsh

from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    diags,
    eye,
    sparray,
    spmatrix,
)

SparseTransitionMatrix = SparseStochMat | csr_matrix | csc_matrix


def set_to_ones(Tcsr: csr_matrix, tol: float = 1e-8) -> None:
    """Replace near-one sparse-matrix entries in place.

    Parameters
    ----------
    Tcsr : :class:`scipy.sparse.csr_matrix`
        Sparse matrix whose stored values are modified in place.
    tol : float, default=1e-8
        Absolute tolerance around one. Stored values ``x`` satisfying
        ``abs(x - 1.0) <= tol`` are replaced by exactly ``1.0``.

    Returns
    -------
    None
        The input matrix is modified in place.
    """
    Tcsr.data[np.abs(Tcsr.data - 1.0) <= tol] = 1.0


def csc_row_normalize(X: csr_matrix | csc_matrix) -> csc_matrix:
    """Return a row-normalized sparse matrix in CSC format.

    Parameters
    ----------
    X : :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`
        Sparse matrix to normalize by row sums. Rows with zero sum are left
        unchanged.

    Returns
    -------
    :class:`scipy.sparse.csc_matrix`
        Row-normalized copy of ``X`` in CSC format.
    """
    X_csr = X.tocsr()

    n_rows = X_csr.shape[0]
    for i in range(n_rows):
        row_sum = X_csr.data[X_csr.indptr[i]:X_csr.indptr[i+1]].sum()
        if row_sum != 0:
            X_csr.data[X_csr.indptr[i]:X_csr.indptr[i+1]] /= row_sum

    return X_csr.tocsc()


def remove_nnz_rowcol(
    L: csr_matrix | csc_matrix,
) -> tuple[csr_matrix | csc_matrix, np.ndarray, int]:
    """Remove rows and columns that are both structurally zero.

    A row and column index is retained when either the corresponding row or
    column contains at least one stored value.

    Parameters
    ----------
    L : :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`
        Square sparse matrix from which structurally zero rows and columns are
        removed.

    Returns
    -------
    L_small : :class:`scipy.sparse.csr_matrix` or \
            :class:`scipy.sparse.csc_matrix`
        Sparse submatrix containing only retained rows and columns.
    nonzero_indices : :class:`numpy.ndarray`
        One-dimensional integer array with the retained row and column indices
        in the original matrix.
    size : int
        Original linear matrix size, equal to ``L.shape[0]``.
    """
    # indices with zero sum row AND col
    nonzerosum_rowcols = ~np.logical_and(L.getnnz(1) == 0,
                                         L.getnnz(0) == 0)

    nonzero_indices, = (nonzerosum_rowcols).nonzero()
    L_small = L[nonzerosum_rowcols][:, nonzerosum_rowcols]

    return (
        L_small,
        nonzero_indices,
        L.shape[0],
    )


def set_to_zeroes(
    T: SparseStochMat | csr_matrix | csc_matrix,
    tol: float | None = 1e-8,
    relative: bool = True,
    use_absolute_value: bool = False,
) -> None:
    """Replace near-zero sparse-matrix entries in place.

    When ``tol`` is not ``None``, stored values close to zero are replaced by
    exact zeros and eliminated from SciPy sparse matrices. For
    :class:`stochmat.SparseStochMat`, the operation is delegated to its own
    ``set_to_zeroes`` method.

    Parameters
    ----------
    T : :class:`stochmat.SparseStochMat`, \
            :class:`scipy.sparse.csr_matrix`, or \
            :class:`scipy.sparse.csc_matrix`
        Sparse matrix whose stored values are modified in place.
    tol : float or None, default=1e-8
        Threshold below which values are set to zero. If ``None``, the function
        returns without modifying ``T``.
    relative : bool, default=True
        If ``True``, scale ``tol`` by the largest absolute stored value before
        thresholding SciPy sparse matrices.
    use_absolute_value : bool, default=False
        If ``True``, threshold values by ``abs(value) <= tol``. If ``False``,
        threshold values by ``value <= tol``.

    Returns
    -------
    None
        The input matrix is modified in place when ``tol`` is not ``None``.

    Raises
    ------
    TypeError
        If ``T`` is not a supported sparse matrix type.
    """
    if tol is not None:
        threshold = float(tol)
        if isinstance(T, SparseStochMat):
            T.set_to_zeroes(threshold, relative=relative)
        elif isinstance(T, (csr_matrix, csc_matrix)):
            data = T.data
            if data.size > 0:
                if relative:
                    # tol = tol*np.abs(T.data).max()
                    # finding the max of the absolute value without making a
                    # copy of the whole array
                    threshold = (
                        threshold
                        * np.abs([data.min(), data.max()]).max()
                    )

                if use_absolute_value:
                    data[np.abs(data) <= threshold] = 0
                else:
                    data[data <= threshold] = 0

                T.eliminate_zeros()
        else:
            raise TypeError("T must be csc,csr or SparseStochMat")


def _threshold_and_row_normalize(
    matrix: SparseTransitionMatrix,
    tol: float | None,
) -> SparseTransitionMatrix:
    """Apply thresholding and row normalization to a sparse matrix.

    Parameters
    ----------
    matrix : SparseStochMat, scipy.sparse.csr_matrix, or scipy.sparse.csc_matrix
        Sparse matrix to process. ``SparseStochMat`` and CSR inputs are
        modified in place. CSC inputs are returned as row-normalized CSC
        copies.
    tol : float or None
        Relative threshold passed to :func:`set_to_zeroes`. If ``None``, only
        row normalization is applied.

    Returns
    -------
    SparseStochMat, scipy.sparse.csr_matrix, or scipy.sparse.csc_matrix
        Thresholded and row-normalized matrix.
    """
    if tol is not None:
        set_to_zeroes(matrix, tol)

    if isinstance(matrix, csc_matrix):
        warnings.warn(
            "CSC transition matrices cannot be row-normalized in place with "
            "stochmat.inplace_csr_row_normalize; returning a row-normalized "
            "CSC copy. Use force_csr=True to avoid this conversion.",
            UserWarning,
            stacklevel=2,
        )
        return csc_row_normalize(matrix)

    inplace_csr_row_normalize(matrix)
    return matrix


def _prepare_inter_transition_matrix(
    inter_matrix: SparseTransitionMatrix,
    *,
    force_csr: bool,
    tol: float | None,
) -> SparseTransitionMatrix:
    """Return a cleaned working copy of an inter-event matrix.

    ``inter_matrix`` is read from ``inter_T``. Since ``inter_T`` is primary
    data, accumulated transition-matrix computations must not mutate it. This
    helper copies the inter-event factor before applying tolerance-based
    sparsification and normalization.

    Parameters
    ----------
    inter_matrix : SparseStochMat, scipy.sparse.csr_matrix, or scipy.sparse.csc_matrix
        Inter-event transition matrix read from ``inter_T``.
    force_csr : bool
        If ``True``, return a CSR matrix. This is required for
        ``SparseStochMat`` inputs because multiplication of CSR by
        ``SparseStochMat`` is not implemented.
    tol : float or None
        Relative tolerance used for sparsification before row normalization.

    Returns
    -------
    SparseStochMat, scipy.sparse.csr_matrix, or scipy.sparse.csc_matrix
        Cleaned working copy. If ``force_csr=True``, the returned matrix is CSR.

    Raises
    ------
    ValueError
        If ``inter_matrix`` is a ``SparseStochMat`` and ``force_csr`` is
        ``False``.
    """
    if isinstance(inter_matrix, SparseStochMat):
        if not force_csr:
            raise ValueError(
                "inter_T[lamda] is a SparseStochMat, but force_csr is False. "
                "Set force_csr=True."
            )
        matrix = inter_matrix.tocsr()
    elif force_csr:
        matrix = inter_matrix.tocsr(copy=True)
    else:
        matrix = inter_matrix.copy()

    return _threshold_and_row_normalize(matrix, tol)


def to_dense(M: ArrayLike | spmatrix | sparray | SparseStochMat) -> np.ndarray:
    """Coerce a sparse or dense matrix-like object to an ndarray.

    Parameters
    ----------
    M : array-like, :class:`scipy.sparse.spmatrix`, \
            :class:`scipy.sparse.sparray`, or \
            :class:`stochmat.SparseStochMat`
        Matrix-like object to convert. NumPy-compatible array-like inputs are
        passed to :func:`numpy.asarray`; SciPy sparse matrices and arrays are
        converted with ``toarray``; and :class:`stochmat.SparseStochMat`
        objects are converted with ``to_full_mat``.

    Returns
    -------
    :class:`numpy.ndarray`
        Dense array representation of ``M``. Existing NumPy arrays are returned
        unchanged.
    """
    if isinstance(M, np.ndarray):
        return M
    if isinstance(M, SparseStochMat):
        M = M.to_full_mat()
    if isinstance(M, (spmatrix, sparray)):
        return M.toarray()
    return np.asarray(M)


def find_spectral_gap(L: csr_matrix | csc_matrix) -> np.ndarray:
    """Compute the spectral gap of a connected random-walk Laplacian.

    Parameters
    ----------
    L : :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`
        Connected random-walk Laplacian matrix. The matrix is converted to CSR
        format internally.

    Returns
    -------
    :class:`numpy.ndarray`
        One-element array returned by :func:`scipy.sparse.linalg.eigsh`,
        containing the eigenvalue nearest zero of the adjusted symmetric
        Laplacian.
    """
    Lcsr = L.tocsr()
    n_nodes = L.shape[0]

    identity = eye(n_nodes, dtype=np.float64, format="csr")

    degs = np.diff((identity - Lcsr).indptr)

    D12 = diags(np.sqrt(degs),
                format="csr")
    Dm12 = diags(1/np.sqrt(degs),
                 format="csr")

    Lsym = D12 @ Lcsr @ Dm12

    # stationary solution
    Pi = np.vstack([degs/degs.sum()]*n_nodes)

    gap = eigsh(Lsym.toarray()-Pi, 1, sigma=0, return_eigenvectors=False)

    return gap
