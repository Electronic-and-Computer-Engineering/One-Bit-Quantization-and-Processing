import numpy as np
import obq  
import scipy
from scipy import signal
import sg, sa, globalTools, filt  
import matplotlib.pyplot as plt

def make_band_aggregator(K, B=2):
    J = int(np.ceil(K / B))
    A = np.zeros((J, K))
    for j in range(J):
        a = j*B; b = min((j+1)*B, K)
        A[j, a:b] = 1.0 / (b - a)
    return A

def iterBlockQDFT(vx, mRange, vW_D, sK, sM, sN_DFT, sHop, sType='grb', verbose=True, nIter=1):
    """
    Iterative block-based one-bit DFT-domain quantization with outer iteration.
    """
    #vx = np.concatenate((np.zeros(sHop), np.ravel(vx)))
    sxLen = len(vx)      
    vWlen = len(vW_D)
        
    sNumBlocks = (sxLen - sM) // sHop + 1
    
    vb = np.zeros(sxLen)
    
    if mRange.ndim == 1:
        mF_D,_   = sg.dftMat(sxLen, sK, mRange) 
        mF_NDFT,_ = sg.dftMat(sN_DFT, sK, mRange) 
        mFW,_   = sg.dftMat(vWlen, sK, mRange)

    elif mRange.ndim > 1:
        for mm in range(mRange.shape[0]):
            FD_m,_ = sg.dftMat(sN_DFT, sK, mRange[mm,:]) 
            if mm == 0:
                F_D = FD_m.copy()
            elif mm > 0:
                F_D = np.vstack([F_D, FD_m])
    
    vWin                = np.hamming(sN_DFT)
    mRIF_NDFT           = np.vstack([mF_NDFT.real, mF_NDFT.imag])
    mRIF_D              = np.vstack([mF_D.real, mF_D.imag])
    vXRI                = mRIF_D @ vx
    vGlobalBlockErr     = np.ones(sNumBlocks)*np.inf
    vBlockErr           = np.ones(sNumBlocks)*np.inf
    vGlobalError        = np.zeros(2*sK)
    #figA, axs          = plt.subplots(1, 2)    
    vEl_hatInit         = np.zeros(2*sK) 
    vX_hatInit          = np.zeros(2*sK) 
    vWin                = np.hamming(sM) #sg.modHamming(sM, (sM//2), 0.6, 0.6, 0.8, 0.8) #np.hamming(sM)   
    
    # for p in range(sNumBlocks):
    #     sStIdx   = p * sHop
    #     sEndIdx  = sStIdx + sM
    #     vx_m     = vx[sStIdx:sEndIdx].copy() 
    #     vWin     = np.hamming(sM)

    #     mRIF_m   = mRIF_D[:,sStIdx:sEndIdx].copy()  
    #     mRIF_mW  = mRIF_m * vWin[np.newaxis, :]
    #     Etest    = Etest.copy() + mRIF_mW @ vx_m
    
    for iIter in range(nIter):
        if verbose:
            print(f"=== Iteration {iIter+1}/{nIter} ===")
            
        for p in range(sNumBlocks):
            if p == 0:
                progressDFTBlock = globalTools.SimpleProgressBar(
                   sNumBlocks, width=40, prefix = "DFT-BlockOpt", fill="█", empty=" ", end=" ✓")
            
            sStIdx   = p * sHop
            sEndIdx  = sStIdx + sM
            
            if sStIdx > (sN_DFT-sM):
                sStL = sStIdx - (sN_DFT-sM)
            else:
                sStL = 0    
          
            if sEndIdx > len(vx):
                if verbose:
                    print(f"Skipping block {p}: exceeds signal length.")
                continue
            
            vx_l     = vx[sStL:sStIdx].copy()
            vb_l     = vb[sStL:sStIdx].copy()  
            vx_m     = vx[sStIdx:sEndIdx].copy() 
            mRIF_m   = mRIF_D[:,sStIdx:sEndIdx].copy()  
            
            if p == 0:
                vEl_hat   = vEl_hatInit.copy()
                vX_hat    = vX_hatInit.copy()
            else:
                
                
                
                mRIF_l    = mRIF_D[:,sStL:sStIdx].copy()
                vEl_hat   = vEl_hat + mRIF_l @ (vx_l - vb_l)
                vX_hat    = vX_hat + mRIF_l @ vb_l
                    

            while True:              
              
                mRIF_mW   = mRIF_m * vWin[np.newaxis, :]
                
                vb_mInit                = vb[sStIdx : sEndIdx].copy()            
                vb_m, sBlockErr, txtOut = obq.OptDFT(vEl_hat, mRIF_mW, vx_m, vb_mInit)
                    
                vGlobalError        = vXRI - (vX_hat + mRIF_m[:,0:sHop] @ vb_m[0:sHop])
                vGlobalBlockErr[p]  = vGlobalError.T @ vGlobalError
                vBlockErr[p]        = sBlockErr
                
                vb[sStIdx: sEndIdx] = vb_m.copy()
                break   
                
            # axs[0].cla()
            # axs[0].plot(vGlobalBlockErr)
            # axs[0].set_title('GlobalError')
            # axs[1].cla()
            # axs[1].plot(vBlockErr)
            # axs[1].set_title('BlockError')

            # plt.pause(0.001)    
                                     
            txtOut += f" BlockErr: {vGlobalBlockErr[p]:.4e}"
            progressDFTBlock.update(p+1, txtOut)
     
    #vb = vb[sHop::]        
    #vb = vb_pad[(sN_DFT-sM):len(vx_pad)].copy()              
    return vb#, vGlobalBlockErr, mErr_k_p