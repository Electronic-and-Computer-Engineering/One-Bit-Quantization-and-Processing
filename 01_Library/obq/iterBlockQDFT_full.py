import numpy as np
import obq  # deine Optimierungsfunktion
from scipy import signal
import sg, sa, globalTools  # Assumes sg.dftMat(N, K) returns (K x N) matrix
import matplotlib.pyplot as plt

def iterBlockQDFT(vx, vRange, sK, sL, sM, sHop, sType='grb', verbose=True, nIter=3):
    """
    Iterative block-based one-bit DFT-domain quantization with outer iteration.
    """
    
    sBlock = sM + sL
    vx = np.concatenate([np.zeros(sL), vx])  # Padding
    sxLen = len(vx)
    sNumBlocks = (sxLen - sBlock) // sHop + 1
    
    vb = np.zeros(sxLen)

    Fp, vOmega = sg.dftMat(sxLen, sK, vRange[0,:])  
    #Fw = np.diag(vW) @ F                  
    #Fw = Fw[0:sK//2, :]
    mRIFw = np.vstack([Fp.real, Fp.imag])
    
    for iIter in range(nIter):
        if verbose:
            print(f"=== Iteration {iIter+1}/{nIter} ===")
            
        for m in range(sNumBlocks):
            if m == 0:
                progressDFTBlock = globalTools.SimpleProgressBar(
                    sNumBlocks, width=40, prefix = "DFT-BlockOpt", fill="█", empty=" ", end=" ✓"
                )
                  
            sStIdx = m * sHop
            sEndIdx = sStIdx + sBlock

            if sEndIdx > sxLen:
                if verbose:
                    print(f"Skipping block {m}: exceeds signal length.")
                continue
            
            vbL     = vb[sStIdx : (sStIdx + sL)]
            vbM_init = vb[(sStIdx + sL) : sEndIdx]
            vxBlock = vx[sStIdx : sEndIdx]
            vxM     = vx[(sStIdx + sL) : sEndIdx]

            mRIFw_L = mRIFw[:, sStIdx : (sStIdx + sL)]
            mRIFw_M = mRIFw[:, (sStIdx + sL) : sEndIdx]

            
            vRIX_tilde = mRIFw[:, sStIdx : sEndIdx] @ vxBlock
            vRIX_hat_L = mRIFw_L @ vbL
            vRIX_t_L   = vRIX_tilde - vRIX_hat_L

            vbM, sBlockErr, txtOut = obq.OptDFT(vRIX_t_L, mRIFw_M, vxM, vbM_init)            
            vb[(sStIdx + sL) : sEndIdx] = vbM
            
            txtOut += f" BlockErr: {sBlockErr:.4e}"
            progressDFTBlock.update(m+1, txtOut)
    
    vb = vb[sL::]
    return vb