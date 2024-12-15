import numpy as np
import sg,misc

def combOptBlock(vx, mW, vE_hat):

    sBSize =    len(vx)
    sNumComb =  2**sBSize
    mComb =     sg.binaryComb(sBSize)
    mComb =     mComb.transpose()*2-1
    vMins =     np.zeros((sNumComb,1)).flatten()    
        
    for k in range(sNumComb):
        veComb   =  mW @ (vx - mComb[:,k]) + vE_hat
        vMins[k] =  np.sum(veComb**2)
    
    sIdxMin =   np.argmin(vMins) 
    vb_out  =   mComb[:,sIdxMin]
    ve_out  =   mW @ (vx - vb_out) + vE_hat   
    
    return vb_out, ve_out