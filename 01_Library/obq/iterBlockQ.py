import numpy as np
#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc

def iterBlockQ(vx, vw, sM, sType):
    """
    Args:
        vx: Input vector.
        mW: Weight matrix.
        vC: Constant/Init vector.
        sL: Number of Decisions
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    swLen = len(vw)
    #swLen_h = int(swLen/2)
    sxLen = len(vx)
    
    ve = np.zeros((sxLen,1)).flatten()         
    vb = np.zeros((sxLen,1)).flatten()   

    vwFull = np.zeros((sxLen,1)).flatten()
    vwFull[0:swLen] = vw
    
    sNumBlocks = int(np.ceil(sxLen/sM))

    #vxPadded = np.pad(vx,(sCutSIdx,(sCutEIdx + additionalPadding)), mode = "reflect")

    vC          = np.zeros((sM,1)).flatten()
    vbBlock     = np.zeros((sM,1)).flatten() 
    veBlock     = np.zeros((sM,1)).flatten()
    veL2Block   = np.zeros((sNumBlocks,1)).flatten()
    vBlockIdx   = np.zeros((sNumBlocks,2))

    if np.mod(sxLen,sM):
        print("vx should be a multiple of sM")
    else:
        mW_0 = np.tril(scLinAlg.toeplitz(vwFull[0:sM]))
        for m in range(sNumBlocks):                                   
            vCe = vC.copy()     #Initialize ve_hat before we actually proceed with the error calculation
            sStIdx = m * sM
            sEndIdx = sStIdx + sM
            vBlockIdx[m,0] = sStIdx
            vBlockIdx[m,1] = sEndIdx
            
            for k in range(m):  #Generation of the vCe
                sRowIdx = sM*(m-k)
                sColIdx = (m-k-1)*sM+1               
                mW_m = scLinAlg.toeplitz(vwFull[sRowIdx:sRowIdx+sM],np.flip(vwFull[sColIdx:sColIdx+sM]))
                vCe += mW_m @ (vx[k*sM:k*sM+sM] - vb[k*sM:k*sM+sM])
            
            if (sType == 'grb'):
                vbBlock, veBlock = obq.OptBlock(vx[m*sM:m*sM+sM], mW_0, vCe)                
            else:
                vbBlock, veBlock = obq.combOptBlock(vx[m*sM:m*sM+sM], mW_0, vCe)    
                
            vb[m*sM:m*sM+sM] = vbBlock
            
            if (m > 0):
                veL2Block[m] = veL2Block[m-1] + np.sum(veBlock**2)
            else:
                veL2Block[m] = np.sum(veBlock**2)
                
            print("BlockNumber: %d, ErrVal: %3.5f" % (m, veL2Block[m]))  
        
    return vb, veL2Block, vBlockIdx