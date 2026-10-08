import numpy as np

def beamBlock(vx, mW, vC_hat, sK=64):
    sM = mW.shape[1]
    mB = np.zeros((1, sM))
    mE = vC_hat[None, :].copy()            # (K, sRows)

    for n in range(sM):
        vCol = mW[:, n]
        mEn  = np.vstack([mE - vCol, mE + vCol]) + vCol*vx[n]

        vNew  = np.sum(mEn[:, :n+1]**2, axis=1)
        sKeep = min(sK, len(vNew))
        vSel  = np.argpartition(vNew, sKeep-1)[:sKeep]

        sNp  = len(mE)
        mB   = mB[vSel % sNp]
        mB[:, n] = np.where(vSel < sNp, 1.0, -1.0)
        mE   = mEn[vSel]

    sBest = int(np.argmin(np.sum(mE**2, axis=1)))
    return mB[sBest].copy(), mE[sBest], float(mE[sBest] @ mE[sBest])