"""
Module: matlab_squareform
Description: Utilities to convert MATLAB-style condensed distance vectors 
             (using column-major ordering) into full symmetric matrices and to extract 
             submatrix representations directly in condensed form.
"""

import numpy as np

def squareformq(cmat):
    """
    Reconstruct an m x m symmetric matrix from the condensed vector cmat,
    matching MATLAB's built-in squareform ordering for i>j (column-major).
    ...
    """
    cmat = np.asarray(cmat).ravel()
    m = int(0.5*(1 + np.sqrt(1 + 8*cmat.size)))
    if m*(m-1)//2 != cmat.size:
        raise ValueError("Length does not match m*(m-1)//2 for any integer m.")
    out = np.zeros((m, m), dtype=cmat.dtype)
    idx = 0
    for col in range(m - 1):
        for row in range(col + 1, m):
            out[row, col] = cmat[idx]
            out[col, row] = cmat[idx]
            idx += 1
    return out

def get_rdm(cmat, indexes):
    """
    Extract the submatrix (in condensed form) for the rows/columns in `indexes`,
    using the same MATLAB column-major ordering for i>j.
    ...
    """
    indexes = np.asarray(indexes)
    m = len(indexes)
    M = int(0.5*(1 + np.sqrt(1 + 8*cmat.size)))
    rdm_sub = np.empty(m*(m-1)//2, dtype=cmat.dtype)

    def full_offset(i, j):
        return (j*(2*M - j - 1)) // 2 + (i - j - 1)

    idx_out = 0
    for sub_col in range(m - 1):
        j_full = indexes[sub_col]
        for sub_row in range(sub_col + 1, m):
            i_full = indexes[sub_row]
            if i_full < j_full:
                i_full, j_full = j_full, i_full
            offset = full_offset(i_full, j_full)
            rdm_sub[idx_out] = cmat[offset]
            idx_out += 1

    return rdm_sub
