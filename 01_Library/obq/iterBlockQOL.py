import numpy as np
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc

def iterBlockQOL(vx, vw, sM, sType, sHop=None):
    """
    Args:
        vx: Input vector.
        mW: Weight matrix.
        vC: Constant/Init vector.
        sM: Block size
        sType: Type of optimization ('grb' or other)
        sHop: Hop size (default: sM-1)
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    swLen = len(vw)
    sxLen = len(vx)
    
    # Set hop size (default is sM-1 if not provided)
    if sHop is None:
        sHop = sM - 1
    
    ve = np.zeros((sxLen,1)).flatten()         
    vb = np.zeros((sxLen,1)).flatten()   

    vwFull = np.zeros((sxLen,1)).flatten()
    vwFull[0:swLen] = vw
    
    sNumBlocks = (sxLen - sM) // sHop + 1  # Adjusted number of blocks based on hop size

    vC          = np.zeros((sM,1)).flatten()
    vbBlock     = np.zeros((sM,1)).flatten() 
    veBlock     = np.zeros((sM,1)).flatten()
    veL2Block   = np.zeros((sNumBlocks,1)).flatten()
    vBlockIdx   = np.zeros((sNumBlocks,2))

    if np.mod(sxLen - sM, sHop):
        print("vx length minus sM should be a multiple of sHop")
    else:
        mW_0 = np.tril(scLinAlg.toeplitz(vwFull[0:sM]))
        for m in range(sNumBlocks):                                   
            vCe = vC.copy()  # Initialize ve_hat before error calculation
            sStIdx = m * sHop  # Start index with hopping
            sEndIdx = sStIdx + sM
            vBlockIdx[m,0] = sStIdx
            vBlockIdx[m,1] = sEndIdx
            
            for k in range(m):  # Generation of vCe
                sRowIdx = sHop * (m - k)
                sColIdx = (m - k - 1) * sHop + 1
                mW_m = scLinAlg.toeplitz(vwFull[sRowIdx:sRowIdx+sM], np.flip(vwFull[sColIdx:sColIdx+sM]))
                vCe += mW_m @ (vx[k*sHop:k*sHop+sM] - vb[k*sHop:k*sHop+sM])
            
            if sType == 'grb':
                vbBlock, veBlock = obq.OptBlock(vx[sStIdx:sEndIdx], mW_0, vCe)
            else:
                vbBlock, veBlock = obq.combOptBlock(vx[sStIdx:sEndIdx], mW_0, vCe)
                
            vb[sStIdx:sEndIdx] = vbBlock
            
            if m > 0:
                veL2Block[m] = veL2Block[m-1] + np.sum(veBlock**2)
            else:
                veL2Block[m] = np.sum(veBlock**2)
                
            print("BlockNumber: %d, ErrVal: %3.5f" % (m, veL2Block[m]))  
        
    return vb, veL2Block, vBlockIdx
