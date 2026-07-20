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

"""Sparse matrix exponential with tolerance-based filtering."""

from typing import Literal

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, eye, find
from scipy.sparse.linalg import norm as sparse_norm

SparseMatrix = csr_matrix | csc_matrix
ScalarNumber = int | float | np.integer | np.floating


def mfp_exp(
    H: SparseMatrix,
    err: ScalarNumber,
    non_norm: Literal[0, 1],
) -> csr_matrix:
    """Evaluate a filtered sparse matrix exponential.

    The Taylor order and scaling depth are selected by
    :func:`tempnet.expm_with_tol.getMN`. Taylor remainders are evaluated with
    :func:`tempnet.expm_with_tol.getfi`, and small entries are removed with
    :func:`tempnet.expm_with_tol.flitoutA2`.

    Parameters
    ----------
    H : :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`
        Square sparse matrix to exponentiate. In temporal-network use, this is
        typically a scaled negative random-walk Laplacian.
    err : float
        Error tolerance used to choose the Taylor order and scaling depth.
    non_norm : int
        Matrix-normality flag. Use ``0`` when ``H`` is normal and ``1`` when
        ``H`` is non-normal.

    Returns
    -------
    :class:`scipy.sparse.csr_matrix`
        Sparse approximation of ``expm(H)`` in CSR format.

    Notes
    -----
    This is a Python translation of Wu Feng's MATLAB implementation of the algorithm
    proposed in "High-performance computation of large sparse matrix
    exponential". Original author: Wu Feng (vonwu@dlut.edu.cn), 2019-08-19.
    Original code: https://www.rocewea.com/1.html
    """

    n_rows = H.shape[0]
    H_norm = sparse_norm(H, 'fro')
    Ih = eye(n_rows, format='csr')

    M, N = getMN(H_norm, err)
    h1 = H_norm / (2**N)
    r0 = getfi(M, h1)

    if non_norm == 0:
        ai = 1 / (N + 1)
    elif non_norm == 1:
        ai = 1 / H_norm

    b0 = ai * r0 / M / np.exp(2 * h1)

    # Compute T_0
    m = 2**N
    H_scaled = H / m
    # NOTE: (j-i-l) copy should not be needed here, operations with S and T
    #               simply rebind the variable. We can avoid the extra copies.
    S = H_scaled.copy()
    T = H_scaled.copy()
    mt = 5

    for i in range(2, M + 1):
        S = S @ (H_scaled / i)
        # Filter S under the condition that norm(S - S1) <= b0
        S, mt = flitoutA2(S, b0, mt)
        sc = S.nnz / (n_rows**2)
        T = T + S

    rn = []
    mt = 1

    for i in range(1, N + 1):
        r0 = 2 * r0 + r0**2
        bi = ai * r0

        T = 2 * T + T @ T

        # Compute the F-norm of Ih + T
        n1 = sparse_norm(T, 'fro')
        dt = T.diagonal()
        n2 = np.dot(dt, dt)
        dt = dt + 1
        n3 = np.dot(dt, dt)
        nt = np.sqrt(n1**2 - n2 + n3)  # nt is the F-norm of Ih + T

        T, mt = flitoutA2(T, bi * nt, mt)
        sc = T.nnz / (n_rows**2)  # The sparsity of T
        rn.append(bi * nt)
        # print(f"[{i}, {sc}]")  # Commented out like in MATLAB

    T = Ih + T

    return T


def flitoutA2(
    A: SparseMatrix,
    eg: ScalarNumber,
    m: ScalarNumber,
) -> tuple[csr_matrix, ScalarNumber]:
    """Filter near-zero entries from a sparse matrix.

    The filtering follows Algorithm 4.1 from "High-performance computing of
    large sparse matrix exponential". If the eliminated matrix is denoted by
    ``B``, the threshold is adjusted until ``norm(B, 'fro') <= eg``.

    Parameters
    ----------
    A : :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`
        Square sparse matrix to filter.
    eg : scalar number
        Positive error bound controlling how much Frobenius norm may be
        removed.
    m : scalar number
        Positive scaling parameter used to derive the elementwise filtering
        threshold. May be a Python or NumPy integer or floating scalar.

    Returns
    -------
    A : :class:`scipy.sparse.csr_matrix`
        Filtered sparse matrix in CSR format.
    m : scalar number
        Updated scaling parameter. The returned value may be non-integer even
        when the input value is an integer.

    Raises
    ------
    ValueError
        If ``eg`` or ``m`` is not positive.

    Notes
    -----
    This is a Python translation of Wu Feng's implementation, written on
    2019-08-19.
    """

    if eg <= 0 or m <= 0:
        raise ValueError("eg and m must be positive")

    filter_tol = eg / 2
    n_rows = A.shape[0]

    ef = filter_tol / m

    # Find non-zero elements
    rows, cols, vals = find(A)

    # Identify elements to keep (above threshold)
    keep_mask = np.abs(vals) > ef

    # Split into kept and removed elements
    kept_rows = rows[keep_mask]
    kept_cols = cols[keep_mask]
    kept_vals = vals[keep_mask]

    removed_rows = rows[~keep_mask]
    removed_cols = cols[~keep_mask]
    removed_vals = vals[~keep_mask]

    # Create matrix of removed elements to compute its norm
    if len(removed_vals) > 0:
        B_removed = csr_matrix(
            (removed_vals, (removed_rows, removed_cols)),
            shape=(n_rows, n_rows),
        )
        b = float(sparse_norm(B_removed, 2))
    else:
        b = 0.0

    # Start building the filtered matrix
    Ia = kept_rows.copy()
    Ja = kept_cols.copy()
    A_vals = kept_vals.copy()

    n = 0
    while b > eg:
        m = b / ef
        ef = filter_tol / m

        # Find which removed elements are now above the new threshold
        newly_kept_mask = np.abs(removed_vals) > ef

        # Add them to the kept set
        Ia = np.concatenate([Ia, removed_rows[newly_kept_mask]])
        Ja = np.concatenate([Ja, removed_cols[newly_kept_mask]])
        A_vals = np.concatenate([A_vals, removed_vals[newly_kept_mask]])

        # Update removed set
        removed_rows = removed_rows[~newly_kept_mask]
        removed_cols = removed_cols[~newly_kept_mask]
        removed_vals = removed_vals[~newly_kept_mask]

        # Recompute norm of remaining removed elements
        if len(removed_vals) > 0:
            B_removed = csr_matrix(
                (removed_vals, (removed_rows, removed_cols)),
                shape=(n_rows, n_rows),
            )
            b = float(sparse_norm(B_removed, 2))
        else:
            b = 0.0

        n = n + 1

    # Construct final filtered matrix
    A = csr_matrix((A_vals, (Ia, Ja)), shape=(n_rows, n_rows))

    return A, m


def getfi(M: int, h: ScalarNumber) -> ScalarNumber:
    """Evaluate the Taylor-series truncation error for ``exp(h)``.

    Parameters
    ----------
    M : int
        Taylor-series order.
    h : scalar number
        Scalar argument of the exponential.

    Returns
    -------
    fi : scalar number
        Relative truncation error of the order-``M`` Taylor series.
    """

    # s0 = h^(M+1) / (M+1)!
    s0 = h
    for i in range(1, M + 1):
        s0 = s0 * h / (i + 1)

    # Compute fi
    i = 0
    si = s0
    f = s0

    while si >= 1e-30:
        si = si * h * (i + M + 1) / (i + 1) / (i + M + 2)
        if np.isinf(si):
            break
        f = f + si
        i = i + 1

    fi = f

    return fi


def getMN(h: ScalarNumber, err: ScalarNumber) -> tuple[int, int]:
    """Select Taylor order and scaling depth from a norm and tolerance.

    Parameters
    ----------
    h : scalar number
        Matrix norm used to determine the scaling depth.
    err : scalar number
        Error tolerance for the Taylor-series approximation.

    Returns
    -------
    M : int
        Selected Taylor-series order.
    N : int
        Selected scaling depth.

    See Also
    --------
    tempnet.expm_with_tol.getfi : Evaluate Taylor truncation errors.
    """

    Nmin = max(int(np.floor(np.log2(h))), 0)
    Nn = 50
    N_vals = []
    M_vals = []

    for i in range(1, 51):
        N_i = Nmin + i
        hi = h / (2**N_i)
        Mj = max(int(np.floor(hi)) - 1, 0)
        rj = 1

        while rj > err:
            Mj = Mj + 1
            r0 = getfi(Mj, hi)
            rj = r0
            for j in range(1, N_i + 1):
                rj = 2 * rj + rj**2

        M_vals.append(Mj)
        N_vals.append(N_i)

    f = np.array(M_vals) * (2.0 ** np.array(N_vals))
    J = np.argmin(f)
    M = M_vals[J]
    N = N_vals[J]

    return M, N
