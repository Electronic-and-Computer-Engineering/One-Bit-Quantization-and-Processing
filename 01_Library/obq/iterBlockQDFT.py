import numpy as np
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc

def iterBlockQDFT(vx, vW, sM, sK, sHop=None, sType = 'grb'):
    """
    Args:
        vx: Input vector.
        vW: Spectral.
        vC: Constant/Init vector.
        sM: Block size
        sType: Type of optimization ('grb' or other)
        sHop: Hop size (default: sM-1)
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    swLen = len(vW)
    sxLen = len(vx)
    
    # Set hop size (default is sM-1 if not provided)
    if sHop is None:
        sHop = sM - 1
    
    ve = np.zeros((sxLen,1)).flatten()         
    vb = np.zeros((sxLen,1)).flatten()   
    
    sNumBlocks = (sxLen - sM) // sHop + 1  # Adjusted number of blocks based on hop size

    vC          = np.zeros((sM,1)).flatten()
    vbBlock     = np.zeros((sM,1)).flatten() 
    veBlock     = np.zeros((sM,1)).flatten()
    veL2Block   = np.zeros((sNumBlocks,1)).flatten()
    vBlockIdx   = np.zeros((sNumBlocks,2))

    for m in range(sNumBlocks):                                   
        vCe = vC.copy()  # Initialize ve_hat before error calculation
        sStIdx  = m * sHop  # Start index with hopping
        sEndIdx = sStIdx + sM - 1
        
        if sType == 'grb':
            vbBlock, veBlock = obq.OptDFT(vx[sStIdx:sEndIdx], vW, sK)
        else:
            vbBlock, veBlock = obq.combOptBlock(vx[sStIdx:sEndIdx], vW, vCe)
            
        vb[sStIdx:sEndIdx] = vbBlock
        
        #if m > 0:
        #    veL2Block[m] = veL2Block[m-1] + np.sum(veBlock**2)
        #else:
        #    veL2Block[m] = np.sum(veBlock**2)
            
        print("BlockNumber: %d, ErrVal: %3.5f" % (m, veL2Block[m]))  
        
    return vb, veL2Block, vBlockIdx
