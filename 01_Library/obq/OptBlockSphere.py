import numpy as np
import sa
from numba import njit

# def sphereDecode(mWtri, vx, vv, sE0=0.0, vbInit=None):
#     """
#     Simple sphere decoding for  min_b || mWtri (vx - vb) + vv ||^2 + sE0

#     Input:
#         mWtri:  M x M, lower triangular (W_△)
#         vx:     input signal (length M)
#         vv:     rotated error of previous blocks, v = Qᵀ c (length M)
#         sE0:    constant E_c (does not change the best b, only the value of E)
#         vbInit: optional start solution in {-1, +1}^M (default: none, E_best = inf)

#     Returning:
#         vb_hat: best one-bit vector in {-1, +1}^M
#         sE:     corresponding error energy E
#         sNodes: number of visited tree nodes
#     """

@njit(cache=True)
def _greedy(mWtri, vx, vv):
    """Start solution: bit by bit, choose the sign that minimizes the current row error."""
    sM = vx.shape[0]
    vb = np.zeros(sM)
    sE = 0.0
    for i in range(sM):
        sBase = vv[i] + mWtri[i, i] * vx[i]              # row i without the term -T_ii b_i
        for j in range(i):
            sBase += mWtri[i, j] * (vx[j] - vb[j])
        vb[i] = 1.0 if sBase >= 0.0 else -1.0            # T_ii > 0  ->  b_i = sign(base)
        sE += (sBase - mWtri[i, i] * vb[i])**2
    return vb, sE


@njit(cache=True)
def _search(mWtri, vx, vv, sE0, vbBest, sEBest):
    """
    Depth-first search, iterative (numba cannot compile recursion efficiently).

    Same tree search as the simple recursive version, with
      - cheaper branch first,
      - lower bound B_i of the rows that are not finished yet:
        the free bits j > i can change row k by at most  S[k, i] = sum_{i<j<=k} |T_kj|,
        so row k keeps at least  max(0, |r_k| - S[k, i])^2  of error.
    """
    sM = vx.shape[0]

    # S[k, i] = sum_{i < j <= k} |T_kj|
    mS = np.zeros((sM, sM))
    for k in range(sM):
        sAcc = 0.0
        for i in range(k, -1, -1):
            mS[k, i] = sAcc
            sAcc += abs(mWtri[k, i])

    # vr[k] = v_k + sum_j T_kj x_j - sum_{fixed j} T_kj b_j   (residual of row k, free bits = 0)
    vr = np.zeros(sM)
    for k in range(sM):
        vr[k] = vv[k]
        for j in range(k + 1):
            vr[k] += mWtri[k, j] * vx[j]

    vb     = np.zeros(sM)
    vE     = np.zeros(sM + 1)                            # vE[i] = E_(i-1), exact partial energy
    vE[0]  = sE0
    mCand  = np.zeros((sM, 2))                           # the two values of b_i, cheaper first
    mECand = np.zeros((sM, 2))                           # exact partial energy E_i
    mLB    = np.zeros((sM, 2))                           # E_i + B_i
    vIdx   = np.zeros(sM, dtype=np.int64)                # next candidate to try on each level
    sNodes = 0
    i      = 0
    bNew   = True

    while True:
        if bNew:                                         # entering level i: evaluate b_i = +1 and -1
            for c in range(2):
                s  = 1.0 if c == 0 else -1.0
                e  = vr[i] - mWtri[i, i] * s             # row error ẽ_i
                sE = vE[i] + e * e                       # E_i = E_(i-1) + ẽ_i²
                sB = 0.0
                for k in range(i + 1, sM):               # lower bound B_i of rows k > i
                    d = abs(vr[k] - mWtri[k, i] * s) - mS[k, i]
                    if d > 0.0:
                        sB += d * d
                mCand[i, c]  = s
                mECand[i, c] = sE
                mLB[i, c]    = sE + sB
            if mLB[i, 1] < mLB[i, 0]:                    # cheaper branch first
                mCand[i, 0],  mCand[i, 1]  = mCand[i, 1],  mCand[i, 0]
                mECand[i, 0], mECand[i, 1] = mECand[i, 1], mECand[i, 0]
                mLB[i, 0],    mLB[i, 1]    = mLB[i, 1],    mLB[i, 0]
            vIdx[i] = 0
            bNew    = False
            sNodes += 1

        if vIdx[i] < 2 and mLB[i, vIdx[i]] < sEBest:     # pruning rule: E_i + B_i >= E_best -> skip
            s  = mCand[i, vIdx[i]]
            sE = mECand[i, vIdx[i]]
            vIdx[i] += 1
            if i == sM - 1:                              # leaf: new best solution
                sEBest = sE
                vbBest[:sM - 1] = vb[:sM - 1]
                vbBest[sM - 1]  = s
            else:                                        # fix b_i and go one level deeper
                vb[i]     = s
                vE[i + 1] = sE
                for k in range(i + 1, sM):
                    vr[k] -= mWtri[k, i] * s
                i   += 1
                bNew = True
        else:                                            # level exhausted: back up, release b_i
            i -= 1
            if i < 0:
                break
            for k in range(i + 1, sM):
                vr[k] += mWtri[k, i] * vb[i]

    return vbBest, sEBest, sNodes


def sphereDecode(mWtri, vx, vv, sE0=0.0, vbInit=None):
    """
    Fast sphere decoding (numba) for  min_b || mWtri (vx - vb) + vv ||^2 + sE0
    Same interface and same result as the simple sphereDecode.

    Input:
        mWtri:  M x M, lower triangular with positive diagonal (W_△)
        vx:     input signal (length M)
        vv:     rotated error of previous blocks, v = Qᵀ c (length M)
        sE0:    constant E_c (does not change the best b, only the value of E)
        vbInit: optional start solution in {-1, +1}^M; default: greedy, row by row

    Returning:
        vb_hat: best one-bit vector in {-1, +1}^M
        sE:     corresponding error energy E
        sNodes: number of visited tree nodes
    """
    mWtri = np.ascontiguousarray(mWtri, dtype=np.float64)
    vx    = np.ascontiguousarray(vx,    dtype=np.float64)
    vv    = np.ascontiguousarray(vv,    dtype=np.float64)

    if vbInit is None:
        vbInit, sEInit = _greedy(mWtri, vx, vv)
    else:
        vbInit = np.array(vbInit, dtype=np.float64)
        ve     = mWtri @ (vx - vbInit) + vv
        sEInit = ve @ ve
    sEInit += sE0

    return _search(mWtri, vx, vv, sE0, vbInit.copy(), sEInit)

def OptBlockSphere(vx, mW, vC_hat, mQ=None, mWtri=None, vbInit=None):
    """
    Universal exact block solver for any filter matrix mW (K x M, K >= M):

        b̂ = argmin_b || mW (vx - b) + vC_hat ||^2

    With the QL decomposition mW = Q W_△ it holds for all b:

        E = || W_△ (vx - b) + v ||^2 + E_c,   v = Qᵀ c,   E_c = ||c||^2 - ||v||^2

    For K = M, E_c = 0 automatically; for an already triangular mW, W_△ = mW.

    Input:
        vx:          input signal (length M)
        mW:          filter matrix (K x M)
        vC_hat:      error of previous blocks (length K)
        mQ, mWtri:   optional precomputed QL decomposition (mW is the same for all blocks)
        vbInit:      optional start solution in {-1, +1}^M

    Returning (same as OptBlock_gram):
        vb_hat:      one-bit vector
        ve_hat:      error vector mW (vx - vb_hat) + vC_hat (length K)
        outTxt:      status text
    """
    if mQ is None or mWtri is None:
        mQ, mWtri = sa.qlDecomp(mW)                     # W = Q W_△

    vv  = mQ.T @ vC_hat                              # v   = Qᵀ c
    sEc = vC_hat @ vC_hat - vv @ vv                  # E_c = ||c||² − ||v||²

    vb_hat, sE, sNodes = sphereDecode(mWtri, vx, vv, sEc, vbInit)

    ve_hat = mW @ (vx - vb_hat) + vC_hat             # true error vector with the original W
    return vb_hat, ve_hat, f"E = {sE:.4f}, {sNodes} nodes"