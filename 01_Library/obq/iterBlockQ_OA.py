import numpy as np
import scipy.linalg as scLinAlg
import obq, globalTools

def iterBlockQ_OA(vx, vw, sM, sType):
    """
    OA-OBBQ: Overlap-Add Block-Based One-Bit Quantization
    For linear-phase FIR filters only. Do NOT use for minimum-phase filters
    → use iterBlockQ instead.

    Args:
        vx:     Input vector.
        vw:     FIR filter impulse response (length L).
        sM:     Block size.
        sType:  'grb' (Gurobi) or brute-force.

    Returns:
        vb:         Quantized one-bit vector.
        veL2Block:  Cumulative L2 error per block (current block only).
        vBlockIdx:  Block start/end indices.
    """

    swLen = len(vw)
    sxLen = len(vx)

    vb        = np.zeros((sxLen, 1)).flatten()
    vwFull    = np.zeros((sxLen, 1)).flatten()
    vwFull[0:swLen] = vw

    sNumBlocks = int(np.ceil(sxLen / sM))
    vbBlock    = np.zeros((sM, 1)).flatten()
    veBlock    = np.zeros((sM, 1)).flatten()
    veL2Block  = np.zeros((sNumBlocks, 1)).flatten()
    vBlockIdx  = np.zeros((sNumBlocks, 2))

    if np.mod(sxLen, sM):
        print("vx should be a multiple of sM")
    else:
        # Linear-phase FIR assumed: dominant block at k* = ⌊(L-1)/(2M)⌋
        sNu = (swLen - 1) // (2 * sM)

        # Build extended filter matrix W_ext  →  shape: ((ν+1)*sM  ×  sM)
        mW_ext = np.vstack([
            np.tril(scLinAlg.toeplitz(vwFull[0:sM]))
            if k == 0 else
            scLinAlg.toeplitz(
                vwFull[k*sM : k*sM + sM],
                np.flip(vwFull[(k-1)*sM+1 : (k-1)*sM+1 + sM]))
            for k in range(sNu + 1)
        ])

        for m in range(sNumBlocks):
            if m == 0:
                progressBlock = globalTools.SimpleProgressBar(
                    sNumBlocks, width=40,
                    prefix="BlockOptimization (OA-OBBQ)",
                    fill="█", empty=" ", end=" ✓")

            # Build extended accumulated error vector  →  length: (ν+1)*sM
            # vCe_ext[j*sM:(j+1)*sM]  =  ĉ_e^(p+j)_past
            vCe_ext = np.zeros((sNu + 1) * sM)
            for j in range(sNu + 1):
                vCe_j = np.zeros(sM)
                for k in range(m):
                    dist    = m + j - k
                    sRowIdx = sM * dist
                    sColIdx = sM * (dist - 1) + 1
                    if sRowIdx >= swLen:
                        continue
                    mW_m = scLinAlg.toeplitz(
                        vwFull[sRowIdx : sRowIdx + sM],
                        np.flip(vwFull[sColIdx : sColIdx + sM]))
                    vCe_j += mW_m @ (vx[k*sM : k*sM + sM]
                                      - vb[k*sM : k*sM + sM])
                vCe_ext[j*sM : (j+1)*sM] = vCe_j

            sStIdx  = m * sM
            sEndIdx = sStIdx + sM
            vBlockIdx[m, 0] = sStIdx
            vBlockIdx[m, 1] = sEndIdx

            if sType == 'grb':
                vbBlock, veBlock, outTxt = obq.OptBlock_OA(
                    vx[m*sM : m*sM + sM], mW_ext, vCe_ext)
            else:
                vbBlock, veBlock = obq.combOptBlock(
                    vx[m*sM : m*sM + sM], mW_ext, vCe_ext)
                outTxt = ""

            vb[m*sM : m*sM + sM] = vbBlock

            # L2 accumulation: current block only (first sM entries of veBlock)
            if m > 0:
                veL2Block[m] = veL2Block[m-1] + np.sum(veBlock[:sM]**2)
            else:
                veL2Block[m] = np.sum(veBlock[:sM]**2)

            progressBlock.update(m + 1, outTxt)

    return vb, veL2Block, vBlockIdx