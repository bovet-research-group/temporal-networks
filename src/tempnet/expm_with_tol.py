import numpy as np
from scipy.sparse import eye, csr_matrix, find
from scipy.sparse.linalg import norm as sparse_norm

def mfp_exp(H, err, non_norm):
    """
    Evaluate the matrix exponential of H using the algorithm proposed in 
    "High-performance computation of large sparse matrix exponential".
    
    Parameters:
    -----------
    H : sparse matrix
        The matrix to be exponentiated
    err : float
        Error tolerance
    non_norm : int
        0 if H is normal matrix, 1 if H is non-normal matrix
    
    Returns:
    --------
    T : sparse matrix
        Matrix exponential of H
    rn : list
        List of error bounds at each iteration
    
    Author: Wu Feng (vonwu@dlut.edu.cn)
    Written on 2019.8.19
    Translated to Python
    """
    
    Nh = H.shape[0]
    H1 = sparse_norm(H, 'fro')
    Ih = eye(Nh, format='csr')
    
    M, N = getMN(H1, err)
    h1 = H1 / (2**N)
    r0 = getfi(M, h1)
    
    if non_norm == 0:
        ai = 1 / (N + 1)
    elif non_norm == 1:
        ai = 1 / sparse_norm(H, 'fro')
    
    b0 = ai * r0 / M / np.exp(2 * h1)
    
    # Compute T_0
    m = 2**N
    H1 = H / m
    S = H1.copy()
    T = H1.copy()
    mt = 5
    
    for i in range(2, M + 1):
        S = S @ (H1 / i)
        # Filter S under the condition that norm(S - S1) <= b0
        S, mt = flitoutA2(S, b0, mt)
        sc = S.nnz / (Nh**2)
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
        sc = T.nnz / (Nh**2)  # The sparsity of T
        rn.append(bi * nt)
        # print(f"[{i}, {sc}]")  # Commented out like in MATLAB
    
    T = Ih + T
    
    return T


def flitoutA2(A, eg, m):
    """
    Filter out the near-zero elements in A to return a sparse matrix A_s 
    using algorithm 4.1 proposed in "High-performance computing of large
    sparse matrix exponential".
    
    If the matrix eliminated after filtering is denoted by B, 
    then b = norm(B, 'fro') <= eg
    
    Parameters:
    -----------
    A : sparse matrix
        Input matrix to filter
    eg : float
        Error tolerance
    m : int
        Scaling parameter
    
    Returns:
    --------
    A : sparse matrix
        Filtered matrix
    m : int
        Updated scaling parameter
    
    Author: Wu Feng
    Written on 2019.8.19
    """
    
    if m == 0:
        m = 1
    
    eg = eg / 2
    Na = A.shape[0]
    
    ef = eg / m
    
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
        B_removed = csr_matrix((removed_vals, (removed_rows, removed_cols)), shape=(Na, Na))
        b = sparse_norm(B_removed, 2)
    else:
        b = 0
    
    # Start building the filtered matrix
    Ia = kept_rows.copy()
    Ja = kept_cols.copy()
    A_vals = kept_vals.copy()
    
    n = 0
    while b > 2 * eg:
        m = b / ef
        ef = eg / m
        
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
            B_removed = csr_matrix((removed_vals, (removed_rows, removed_cols)), shape=(Na, Na))
            b = sparse_norm(B_removed, 2)
        else:
            b = 0
        
        n = n + 1
    
    # Construct final filtered matrix
    A = csr_matrix((A_vals, (Ia, Ja)), shape=(Na, Na))
    
    return A, m


def getfi(M, h):
    """
    Evaluate the relative error of the M-order Taylor series of exp(h).
    
    Parameters:
    -----------
    M : int
        Order of Taylor series
    h : float
        Input value
    
    Returns:
    --------
    fi : float
        Relative error
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


def getMN(h, err):
    """
    In terms of h and the error tolerance err, select M and N.
    
    Parameters:
    -----------
    h : float
        Matrix norm
    err : float
        Error tolerance
    
    Returns:
    --------
    M : int
        Selected M parameter
    N : int
        Selected N parameter
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
