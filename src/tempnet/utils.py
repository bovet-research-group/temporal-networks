import numpy as np
from stochmat import inplace_csr_row_normalize, SparseStochMat
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc, vstack

def set_to_ones(Tcsr, tol=1e-8):
    """In-place replacement of ones in sparse matrix within the tolerence.

    Replace values within a tolerance to one with actual ones.
    """
    Tcsr.data[np.abs(Tcsr.data - 1) <= tol] = 1

def _prep(M, force_csr=False, tol=None):
            M = M.tocsr() if force_csr else M
            if tol is not None:
                set_to_zeroes(M, tol)
                inplace_csr_row_normalize(M)
            return M

def csc_row_normalize(X):
    """Row normalize scipy sparse csc matrices.
    returns a copy of X row-normalized and in CSC format.
    """
    X = X.tocsr()

    for i in range(X.shape[0]):
        row_sum = X.data[X.indptr[i]:X.indptr[i+1]].sum()
        if row_sum != 0:
            X.data[X.indptr[i]:X.indptr[i+1]] /= row_sum

    return X.tocsc()



def remove_nnz_rowcol(L):
    """CSC or CSR matrix with removed zero row and columns

    This also returns an array of the indices of rows/columns with non-zero
    values and the (linear) size of L.

    Returns
    -------
    L_small, nonzero_indices, size

    """
    # indicies with zero sum row AND col
    nonzerosum_rowcols = ~np.logical_and(L.getnnz(1) == 0,
                                         L.getnnz(0) == 0)

    nonzero_indices, = (nonzerosum_rowcols).nonzero()

    return (
        L[nonzerosum_rowcols][:, nonzerosum_rowcols],
        nonzero_indices,
        L.shape[0]
    )



def set_to_zeroes(Tcsr, tol=1e-8, relative=True, use_absolute_value=False):
    """In-place replacement of zeroes in sparse matrix within a tolerance.

    Replace values that are, within the tolerence, close to zero with actual
    zeroes.

    If tol is None, does nothing
    """
    if tol is not None:
        if isinstance(Tcsr, SparseStochMat):
            Tcsr.set_to_zeroes(tol, relative=relative)
        elif isinstance(Tcsr, (csr_matrix, csc_matrix)):
            if Tcsr.data.size > 0:
                if relative:
                    # tol = tol*np.abs(Tcsr.data).max()
                    # finding the max of the absolute value without making a
                    # copy of the whole array
                    tol = tol*np.abs([Tcsr.data.min(), Tcsr.data.max()]).max()

                if use_absolute_value:
                    Tcsr.data[np.abs(Tcsr.data) <= tol] = 0
                else:
                    Tcsr.data[Tcsr.data <= tol] = 0

                Tcsr.eliminate_zeros()
        else:
            raise TypeError("Tcsr must be csc,csr or SparseStochMat")
def _dense(M):
        """Coerce sparse, SparseStochMat, or dense matrix-like to a 2D ndarray."""
        # SparseStochMat (from the stochmat package)
        if hasattr(M, "to_full_mat"):
            M = M.to_full_mat()
        # scipy sparse
        if hasattr(M, "toarray"):
            return M.toarray()
        return np.asarray(M)

def _csr(M):
    """Coerce SparseStochMat, scipy sparse, or dense matrix-like to a CSR matrix."""
    if hasattr(M, "to_full_mat"):        # SparseStochMat (stochmat package)
        M = M.to_full_mat()
    if hasattr(M, "tocsr"):              # scipy sparse
        return M.tocsr()
    return csr_matrix(np.asarray(M))     # dense

def find_spectral_gap(L):
    """L is assummed to be connected"""
    Lcsr = L.tocsr()

    I = eye(L.shape[0],
            dtype=np.float64,
            format="csr")

    degs = np.diff((I-Lcsr).indptr)

    D12 = diags(np.sqrt(degs),
                format="csr")
    Dm12 = diags(1/np.sqrt(degs),
                 format="csr")

    Lsym = D12 @ Lcsr @ Dm12

    # stationary solution
    Pi = np.vstack([degs/degs.sum()]*L.shape[0])

    gap = eigsh(Lsym.toarray()-Pi, 1, sigma=0, return_eigenvectors=False)

    return gap

