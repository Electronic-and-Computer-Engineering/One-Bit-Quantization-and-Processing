import numpy as np
import scipy.linalg as scLinAlg
import obq, globalTools, sa


def iterBlockQ(vx, vw, sM, sPhase = 'lin', sK = None, sEps = 0.05,
               sType = 'grb', bQL = None, bSilent = False):
    """
    OA-OBBQ: Overlap-Add Block-Based One-Bit Quantization

    This implementation works for both, minimum and linear FIR filters,
    the decision is tackled by setting sPhase = 'min' and reproduces ISCAS25
    exactly.

    Per block m the extended cost

        E_ext = || W_ext (x^(m) - b^(m)) + vCe_ext ||^2

    is minimised, with W_ext the first sK rows of the first block column of W
    and vCe_ext the contribution of all previously decided samples.

    Input-Arguments:
        vx:     Input vector (length N, multiple of sM).
        vw:     FIR filter impulse response (length L).
        sM:     Block size.
        sPhase: 'min' -> sK = sM             (one block, no overlap)
                'lin' -> sK = n_lo + sM      (energy rule, see below)
        sK:     Override for the number of rows of W_ext, sM <= sK <= L+sM-1.
                Need NOT be a multiple of sM. Only for sweeping over sK.
        sEps:   Neglected-energy fraction of the 'lin' rule.
        sType:  'grb' (Gurobi), 'tabu' (Tabu-Search), 'sphere', or brute-force.
        bQL:    Hand the QL factors to the solver instead of (W_ext, vCe_ext).
                Always on for 'sphere', which needs the triangular factor;
                off by default otherwise. Kept off for 'grb' so the Gram form
                keeps seeing the same objective value as before: MIPGap is
                relative and the constant ||vCe_ext||^2 - ||vv||^2 would
                shift it.

    Returning:
        vb:         Quantized one-bit vector.
        veL2Block:  Cumulative L2 error per block (current block only).
    """

    swLen = len(vw)
    sxLen = len(vx)

    if np.mod(sxLen, sM):
        raise ValueError("len(vx) = %d is not a multiple of sM = %d" % (sxLen, sM))

    if sType == 'sphere':
        bQL = True                      # sphere decoding needs the QL factor
    elif bQL is None:
        bQL = False

    # ---- number of rows of W_ext ----------------------------------------
    #  Row r weights the sM bits of the current block with the energy
    #      gamma(r) = sum_{n=r-M+1}^{r} w_n^2,
    #  and the sM undetermined bits of the NEXT block, whose contribution the
    #  model sets to zero, with
    #      pi(r) = sum_{n=r-2M+1}^{r-M} w_n^2 = gamma(r - M).
    #  Same function of the filter energy, shifted by sM: the price of a row is
    #  the benefit of the row sM positions earlier. Rows are admitted while that
    #  price stays below sEps of the filter energy, hence sK = n_lo + sM with
    #      n_lo = max{ n : Phi(n) <= sEps ||w||^2 },  Phi(r) = sum_{n<=r} w_n^2
    #  the energy-based effective length of vw [Laakso & Valimaki, T-IM 1999].
    #
    #  'min': energy at the front, n_lo = -1  ->  sK = sM
    #  'lin': energy at c = (L-1)/2           ->  sK = n_lo + sM

    if sK is None:
        if sPhase == 'min':
            sK = sM
        else:
            vPhi = np.cumsum(vw**2) / np.sum(vw**2)
            sNlo = int(np.searchsorted(vPhi, sEps, side = 'right')) - 1
            sK   = sNlo + sM
    sK = int(np.clip(sK, sM, swLen + sM - 1))

    # ---- filter matrices, built ONCE ------------------------------------
    #  mWfull[i,j] = w_{i-j},  i < L+M-1, j < M     (whole tail of one block)
    #  mW_ext      = first sK rows                  (the model)
    #  mWrow[i,p]  = w_{(L-1)-p+i}                  (all past samples at once)
    #
    #  mWrow collapses the sum over past blocks into one constant matrix: the
    #  offsets are fixed relative to the current block, so the contribution of
    #  the last L-1 differences is a single matvec.

    vwPad  = np.zeros(swLen + sM - 1)
    vwPad[0:swLen] = vw

    mWfull = scLinAlg.toeplitz(vwPad, np.concatenate(([vw[0]], np.zeros(sM - 1))))
    mW_ext = mWfull[0:sK, :]
    mWrow  = scLinAlg.toeplitz(np.concatenate(([vw[-1]], np.zeros(sK - 1))),
                               np.flip(vw[1:]))

    # ---- QL, once for all blocks ----------------------------------------
    #  mW_ext = mQ @ mWtri  ->  E_ext = ||mWtri d + vv||^2 + sEc, vv = mQ.T c
    #  The solver then works on an (sM x sM) triangular problem regardless of
    #  sK; sK only enters through the O(sK sM) product mQ.T @ vCe_ext, once per
    #  block. Signature unchanged: the solver gets (mWtri, vv) in place of
    #  (mW_ext, vCe_ext), so obq does not need to know about the factorisation.

    if bQL:
        mQ, mWtri   = sa.qlDecomp(mW_ext)
        mQt         = mQ.T
        mWuse       = mWtri
    else:
        mQt, mWuse = None, mW_ext

    # ---- preallocation ---------------------------------------------------
    sNumBlocks = sxLen // sM
    vb         = np.zeros(sxLen)
    vdPad      = np.zeros(swLen - 1 + sxLen)   # vdPad[L-1+n] = d_n, 0 for n < 0
    veL2Block  = np.zeros(sNumBlocks)
    outTxt     = ""                     # Just for the statusBar

    for m in range(sNumBlocks):
        if (m == 0) & (bSilent == False):
            progressBlock = globalTools.SimpleProgressBar(sNumBlocks, width=40, prefix = "BlockOptimization", fill="█", empty=" ", end=" ✓")

        sStIdx  = m * sM
        sEndIdx = sStIdx + sM

        # ---- extended accumulated error vector, length sK ---------------
        #      vCe_ext[i] = sum_{n < m*sM} w_{m*sM + i - n} d_n

        vCe_ext = mWrow @ vdPad[sStIdx : sStIdx + swLen - 1]

        if bQL:
            vvUse = mQt @ vCe_ext
            sEc   = float(vCe_ext @ vCe_ext - vvUse @ vvUse)
        else:
            vvUse, sEc = vCe_ext, 0.0

        if sType == 'grb':
            vbBlock, _, outTxt = obq.OptBlock(vx[sStIdx:sEndIdx], mWuse, vvUse)
        elif sType == 'tabu':
            vbBlock, _, outTxt = obq.OptBlockTabu(vx[sStIdx:sEndIdx], mWuse, vvUse, sNumReads=200, sTimeout=10)
        elif sType == 'sphere':
            vbBlock, _, outTxt = obq.OptBlockSphere(vx[sStIdx:sEndIdx], mWuse, vvUse, sEc)
        else:
            vbBlock, _, _, _ = obq.combOptBlock_dec(vx[sStIdx:sEndIdx], mWuse, vvUse)
            outTxt = ""

        vb[sStIdx:sEndIdx] = vbBlock
        vdBlock            = vx[sStIdx:sEndIdx] - vbBlock
        vdPad[swLen - 1 + sStIdx : swLen - 1 + sEndIdx] = vdBlock

        # ---- error vector from the ORIGINAL mW_ext, never from the factor -
        veBlock = mW_ext @ vdBlock + vCe_ext

        # L2 accumulation: current block only (first sM entries of veBlock)
        if m > 0:
            veL2Block[m] = veL2Block[m-1] + np.sum(veBlock[:sM]**2)
        else:
            veL2Block[m] = np.sum(veBlock[:sM]**2)

        if (bSilent == False):
            progressBlock.update(m+1, outTxt)

    return vb, veL2Block