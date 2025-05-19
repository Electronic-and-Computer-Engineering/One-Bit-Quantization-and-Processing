import numpy as np
import obq  # deine Optimierungsfunktion
from scipy import signal
import sg, sa, globalTools  # Assumes sg.dftMat(N, K) returns (K x N) matrix
import matplotlib.pyplot as plt



def iterBlockQDFT(vx, vW, sM, sL, sHop=None, sType='grb', verbose=True):
    """
    Iterative block-based one-bit DFT-domain quantization.

    Args:
        vx: Input signal (1D array), length N
        vW: Spectral weights (e.g., for frequency filtering), length K
        sM: Length of the optimizable segment per block
        sL: Length of the locked (fixed) segment per block
        sHop: Hop size between blocks (default = sL)
        sType: Optimization type ('grb', 'other')
        verbose: If True, print block-wise info

    Returns:
        vb_full: Quantized signal (one-bit), length N
    """
    vW = vW[::(len(vW)//2)]
    sK = len(vW)
    vx = np.pad(vx, mode='constant')  # pad at start for first block
    sxLen = len(vx)

    if sHop is None:
        sHop = sL

    sBlockLen = sL + sM
    sNumBlocks = (sxLen - sBlockLen) // sHop + 1
    
    vBlockErr = np.zeros((sNumBlocks,1)).flatten()
    vxMean    = np.zeros((sNumBlocks,1)).flatten()
    vb = np.zeros(sxLen)
    ###### Fix Params ######
    # DFT matrix (K x N)
    F = sg.dftMat(sBlockLen, sK)              # shape: (sK x sBlockLen)
    Fw = np.diag(vW) @ F                      # shape: (sK x sBlockLen)
    ########################
    np.save('analysis/mF.npy', F)
    np.save('analysis/mFw.npy', Fw)
    np.save('analysis/vW.npy', vW)
    np.save('analysis/sL.npy', sL)
    np.save('analysis/sK.npy', sK)
    np.save('analysis/sM.npy', sM)
    np.save('analysis/sHop.npy', sHop)
    
    sFs = np.load('saves/sFs.npy')
    np.save('analysis/sFs.npy', sFs)
    
    for m in range(sNumBlocks):
        if m == 0:
            progressDFTBlock = globalTools.SimpleProgressBar(sNumBlocks, width=40, prefix = "BlockOptimization (ISCAS25)", fill="█", empty=" ", end=" ✓")
              
        sStIdx = m * sHop
        sEndIdx = sStIdx + sBlockLen

        if sEndIdx > sxLen:
            if verbose:
                print(f"Skipping block {m}: exceeds signal length.")
            continue

        vbL  = vb[sStIdx : sStIdx + sL]
        vx_block = vx[sStIdx : sEndIdx] - np.mean(vx[sStIdx : sEndIdx])

        if sType == 'grb':
            if verbose:
                print(f"Block {m+1:03d}/{sNumBlocks}: Start={sStIdx}, End={sEndIdx} ==> ", end='')
            vbM, sBlockErr = obq.OptDFT(vx_block, vW, sK, vbL, sM)
        else:
            vbM = obq.combOptBlock(vx_block, vW, np.zeros_like(vbL))

        vb[sStIdx + sL : sEndIdx] = vbM
        
        vBlockErr[m] = sBlockErr
        vxMean[m]    = np.mean(vx_block)
        block_data = {
            "x_block": vx_block,
            "b_full": vb[sStIdx : sEndIdx],
            "vBlockError": vBlockErr,
            "vxMean": vxMean
            }
        np.savez(f"analysis/anBlock/block_{m:03d}.npz", **block_data)
    
    vb = vb[sL:]
    return vb