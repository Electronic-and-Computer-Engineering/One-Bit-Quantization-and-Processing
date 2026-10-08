import numpy as np
import globalTools as gT
import sg

def combOptBlock_dec(vx, mW, vC_hat, sDepthMax=4, sStep=1, vbInit=None, bPlot=False):
    """
    Block optimisation by a cascade of growing neighbourhoods (VND).

    Depth runs 1 .. sDepthMax, each pass starts at the fixed point of the
    previous one. Contiguous windows, sliding with sStep. Once the cascade
    stalls, swapCloseBits solves exactly over the tightest bits; if that
    finds something, the cascade runs again.

    Returns: vb_hat, ve, sE, sStallIdx
    """

    sBSize = len(vx)
    vb_hat = sg.sgn0(vx) if vbInit is None else vbInit.copy()

    ve = mW @ (vx - vb_hat) + vC_hat
    sE = ve @ ve

    dPlot     = gT.plotInit(bPlot)
    gT.plotAdd(dPlot, sE)
    sStallIdx = None

    for sDepth in range(1, sDepthMax + 1):

        gT.plotDepth(dPlot, sDepth)
        mComb = sg.binaryComb(sDepth).T * 2.0 - 1.0

        while True:
            sE_old = sE

            for sBIdx in range(0, sBSize, sStep):
                sLen    = min(sDepth, sBSize - sBIdx)
                sBEnd   = sBIdx + sLen
                mWSlice = mW[:, sBIdx:sBEnd]
                mCombL  = mComb[sDepth-sLen:, :2**sLen]

                veRest = ve + mWSlice @ vb_hat[sBIdx:sBEnd]
                mERest = veRest[:, None] - mWSlice @ mCombL
                vMins  = np.sum(mERest**2, axis=0)

                sIdxMin = np.argmin(vMins)
                sENew   = vMins[sIdxMin]

                if sENew < sE - 1e-12*sE:
                    sStallIdx = None
                elif sStallIdx is None:
                    sStallIdx = sBIdx

                vb_hat[sBIdx:sBEnd] = mCombL[:, sIdxMin]
                ve[:]               = mERest[:, sIdxMin]
                sE                  = sENew

                gT.plotAdd(dPlot, sE)

            if sE_old - sE <= 1e-12 * sE_old:
                break

    gT.plotClose(dPlot)
    return vb_hat, ve, sE, sStallIdx