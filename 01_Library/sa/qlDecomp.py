import numpy as np


def qlDecomp(mW):
    """
    QL decomposition  mW = mQ @ mWtri

    Input:
        mW:     K x M, K >= M, full column rank

    Returning:
        mQ:     K x M, mQ.T @ mQ = I        (rotation)
        mWtri:  M x M, lower triangular with positive diagonal (W_△)

    Computed with Householder reflections (numpy QR), which keep mQ orthogonal
    to machine precision even for badly conditioned filter matrices.
    QL is obtained from QR by reversing the column order:
        mW[:, ::-1] = Q R   ->   mW = Q[:, ::-1] @ flip(R)
    """
    mQ, mR = np.linalg.qr(np.flip(mW, axis=1))      # QR of the column-reversed matrix
    mWtri  = np.flip(mR)                             # reverse rows and columns -> lower triangular
    mQ     = np.flip(mQ, axis=1)                     # matching columns of Q

    vSign  = np.sign(np.diag(mWtri))                 # make the diagonal positive
    vSign[vSign == 0] = 1.0
    mWtri  = vSign[:, None] * mWtri
    mQ     = mQ * vSign[None, :]
    return mQ, mWtri