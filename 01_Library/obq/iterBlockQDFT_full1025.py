import numpy as np
import obq  
import sg, sa, globalTools

def iterBlockQDFT(vx, mRange, vw_D, sK, sM, sN_DFT, sHop, sType='grb', verbose=True, nIter=1):
    """
    Iterative block-based one-bit DFT-domain quantization with outer iteration.
    """

    sxLen               = len(vx)      
    vWlen               = len(vw_D)
    vEl_hatInit         = np.zeros(2*sK) 
    vX_hatInit          = np.zeros(2*sK)
    
    strWinType          = 'hann'
    sPad                = sM // 4
    sxLenPad            = sxLen + 2*sPad
    vx_pad              = np.pad(vx, (sPad, sPad), mode = 'constant', constant_values=(0, 0))
    
    mBlocks,_ ,_        = sa.OLA_analysis(vx, sM, sHop, strWinType, sPad, False)
    mB                  = np.zeros((mBlocks.shape[0],mBlocks.shape[1]))
    mD                  = np.zeros((mBlocks.shape[0],mBlocks.shape[1]))
    
    vb = np.zeros(sxLenPad)
    
    if mRange.ndim == 1:
        mF_D,_    = sg.dftMat(sxLenPad, sK, mRange) 
        mF_NDFT,_ = sg.dftMat(sN_DFT, sK, mRange)
        mF_W,_    = sg.dftMat(vWlen, sK, mRange) 

    elif mRange.ndim > 1:
        for mm in range(mRange.shape[0]):
            FD_m,_ = sg.dftMat(sN_DFT, sK, mRange[mm,:]) 
            if mm == 0:
                F_D = FD_m.copy()
            elif mm > 0:
                F_D = np.vstack([F_D, FD_m])
    
    mRIF_D              = np.vstack([mF_D.real, mF_D.imag])
    vW_D                = mF_W @ vw_D
    vW_D                = sa.setTol(vW_D.copy(), 1e-12, 0.0, 0.0, 'Snap')
    mF_WD               = np.diag(vW_D) @ mF_D
    mRIF_WD             = np.vstack([mF_WD.real, mF_WD.imag])
    vXRI                = mF_D @ vx_pad
    
    vGlobalBlockErr     = np.ones(mBlocks.shape[1])*np.inf
    vBlockErr           = np.ones(mBlocks.shape[1])*np.inf
    vGlobalError        = np.zeros(2*sK)

    for p in range(mBlocks.shape[1]):
        if p == 0:
            progressDFTBlock = globalTools.SimpleProgressBar(
               mBlocks.shape[1], width=40, prefix = "DFT-BlockOpt", fill="█", empty="[ ]", end="[✓]")
        
        vd_l     = sa.OLA_synth(mD, sM, sHop, strWinType, sPad, False)
        sStIdx   = p * sHop
        sEndIdx  = sStIdx + sM
        vEl_hat  = mRIF_D[:,0:sStIdx].copy() @ vd_l[:,0:sStIdx]
        
        
        vb_m, sBlockErr, txtOut = obq.OptDFT(vEl_hat, mRIF_mW, vx_m, vb_mInit)
              
        vx_m     = mBlocks[:,p].copy() 
        mRIF_m   = mRIF_D[:,sStIdx:sEndIdx].copy()  
       
        if p == 0:
            vEl_hat   = vEl_hatInit.copy()
            vX_hat    = vX_hatInit.copy()
        else:
            
            mRIF_l    = mRIF_D[:,sStL:sStIdx].copy()
            vEl_hat   = vEl_hat + mRIF_l @ (vx_l - vb_l)
            vX_hat    = vX_hat + mRIF_l @ vb_l
                                 
        #txtOut += f" BlockErr: {vGlobalBlockErr[p]:.4e}"
        progressDFTBlock.update(p+1, '')
                
    return vb#, vGlobalBlockErr, mErr_k_p