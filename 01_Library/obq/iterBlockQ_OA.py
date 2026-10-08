import numpy as np
import scipy.linalg as scLinAlg
import obq, globalTools

def iterBlockQ_OA(vx, vw, sM, sPhase = 'lin', sNumIter = 1, sK = None, sType = 'grb', bSilent = False):
    """
    OA-OBBQ: Overlap-Add Block-Based One-Bit Quantization
    
    This implementation works for both, minimum and linear FIR filters, 
    the decision is tackled by setting sPhase = 'min' and reproduces ISCAS25 exactly

    Input-Arguments:
        vx:     Input vector (length N, multiple of sM).
        vw:     FIR filter impulse response (length L).
        sM:     Block size.
        sPhase: 'min' -> sK = sM            (one block, no overlap)
                'lin' -> sK = (L-1)/2 + sM  (overlap up to the symmetry centre)
        sK:     Override for the number of rows of W_ext, sM <= sK <= L+sM-1.
                Need NOT be a multiple of sM. Only for sweeping over sK.
        sType:  'grb' (Gurobi), 'tabu' (Tabu-Search), or brute-force.

    Returning:
        vb:         Quantized one-bit vector.
        veL2Block:  Cumulative L2 error per block (current block only).
        vBlockIdx:  Block start/end indices.
    """

    swLen = len(vw)
    sxLen = len(vx)
    
    if np.mod(sxLen, sM):
        raise ValueError("len(vx) = %d is not a multiple of sM = %d" % (sxLen, sM))

    # ---- block matrices W^(0) ... W^(kMax), built ONCE -------------------
    #      W^(k)_ij = w_{kM+i-j}                                   
    kMax            = (swLen + sM - 2) // sM
    vwFull          = np.zeros((kMax + 2) * sM)
    vwFull[0:swLen] = vw
    # All Blocks are build once, in order to NOT recompute them all the time
    # lW means "list" of W's
    
    lW = [np.tril(scLinAlg.toeplitz(vwFull[0:sM])) if k == 0 else
          scLinAlg.toeplitz(vwFull[k*sM : k*sM + sM],
                            np.flip(vwFull[(k-1)*sM+1 : (k-1)*sM+1 + sM]))
          for k in range(kMax + 1)]  
    
    # ---- number of rows of W_ext ----------------------------------------
    #  Row r of the stacked matrix sees taps r-(M-1) ... r. 
    #  Rows beyond the last significant tap carry no information about the block being decided,
    #  only the price of the zeroed future terms.
    #
    #  'min': energy at the front, c ~ 0   -> sK = sM        (nu = 0)
    #  'lin': energy at c = (L-1)/2        -> sK = (L-1)/2 + sM
    
    sNlo = int(np.searchsorted(np.cumsum(vw**2)/np.sum(vw**2), 0.05))
    sK   = sNlo + sM
    
    mWstack = np.vstack(lW)                                  # ((kMax+1)*sM, sM)
    mWstack = np.vstack([mWstack, np.zeros((sK, sM))])       # Nullen zum Ueberhang
    
    mW_ext = np.vstack(lW)[0:sK, :]          # (sK x sM)            
    mWfull = np.vstack(lW)                   # (sKfull, sM), ungekuerzt
    sKfull = mWfull.shape[0]                 # (kMax+1)*sM
    
    # ---- preallocation ---------------------------------------------------
    sNumBlocks = int(np.ceil(sxLen / sM))
    vb         = np.zeros(sxLen)
    veL2Block  = np.zeros(sNumBlocks)
    vBlockIdx  = np.zeros((sNumBlocks, 2))
    outTxt     = ""                     # Just for the statusBar

    for sIterIdx in range(sNumIter):
        
        vE_g = np.zeros(sxLen + sKfull)
        vE_g[:sxLen] = np.convolve(vw, (vx - vb))[:sxLen]
        
        for m in range(sNumBlocks):
            if (m == 0) & (bSilent == False):
                progressBlock = globalTools.SimpleProgressBar(sNumBlocks, width=40, prefix = "BlockOptimization", fill="█", empty=" ", end=" ✓")
                            
            # ---- extended accumulated error vector, length sK ---------------
            #      vCe_ext[j*sM+i] = sum_{k<m} W^(m+j-k)[i,:] d^(k)
            #
            #  k only contributes while dist = m+j-k <= kMax, hence the lower
            #  loop bound; no guard needed. Last row-block is truncated to nR.
            
            sStIdx  = m * sM
            sEndIdx = sStIdx + sM
            vBlockIdx[m, 0] = sStIdx
            vBlockIdx[m, 1] = sEndIdx
            
            if sIterIdx == 0:
                vCe_ext = np.zeros(sK)
                for k in range(max(0, m - kMax), m):
                    sD       = (m - k) * sM
                    vCe_ext += mWstack[sD:sD+sK, :] @ (vx[k*sM : k*sM+sM]
                                                       - vb[k*sM : k*sM+sM])
                vbInit = None
                mWuse  = mW_ext
                sKuse  = sK
            else:
                vCe_ext = vE_g[sStIdx:sStIdx+sKfull] - mWfull @ (
                              vx[sStIdx:sEndIdx] - vb[sStIdx:sEndIdx])
                vbInit  = vb[sStIdx:sEndIdx]
                mWuse   = mWfull
                sKuse   = sKfull
    
            if sType == 'grb':
                vbBlock, veBlock, outTxt = obq.OptBlock_gram(vx[m*sM : m*sM + sM], mW_ext, vCe_ext)
            elif sType == 'tabu':
                vbBlock, veBlock, outTxt = obq.OptBlockTabu(vx[m*sM : m*sM + sM], mW_ext, vCe_ext, sNumReads=200, sTimeout=10)  
            elif sType == 'sphere':          
                vbBlock, veBlock, outTxt = obq.OptBlockSphere(vx[m*sM : m*sM + sM], mWuse, vCe_ext, vbInit = vbInit)
            else:
                vbBlock, veBlock,_ ,_ = obq.combOptBlock_dec(vx[m*sM : m*sM + sM], mWuse, vCe_ext, vbInit = vbInit)
                outTxt = ""
    
            vb[m*sM : m*sM + sM] = vbBlock
            if sIterIdx > 0:
                vE_g[sStIdx:sStIdx+sKuse] = veBlock
    
            # L2 accumulation: current block only (first sM entries of veBlock)
            if m > 0:
                veL2Block[m] = veL2Block[m-1] + np.sum(veBlock[:sM]**2)
            else:
                veL2Block[m] = np.sum(veBlock[:sM]**2)
    
            if (bSilent == False):
                progressBlock.update(m+1, outTxt)

    return vb, veL2Block, vBlockIdx