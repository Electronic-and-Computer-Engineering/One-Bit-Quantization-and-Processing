import numpy as np

def iterSequQ(vx, mW, se_init):
    """
    Args:
        vx: Input vector.
        mW: Weight matrix.
        se_init: Constant/Init error value.
        sL: Number of Decisions
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    sRWLen = len(mW[:,1])
    sxLen = len(vx)
    ve = np.zeros((sRWLen,1)).flatten()         
    vb = np.zeros((sRWLen,1)).flatten()   
    ve_hat = np.zeros((sRWLen,1)).flatten() 

    for sn in range(sxLen):
        se_hat = se_init
        for k in range(sn):
            se_hat += mW[sn,k] * (vx[k] - vb[k])

        ve_hat[sn] = se_hat
        vb[sn] = 1                                   
        se_p = mW[sn, :] @ vx - mW[sn, :] @ vb
        vb[sn] = -1
        se_n = mW[sn, :] @ vx - mW[sn, :] @ vb
        
        if abs(se_p)**2 <= abs(se_n)**2:
            vb[sn] = 1
            ve[sn] = se_p
        else:
            vb[sn] = -1
            ve[sn] = se_n
                   
    return vb, ve, ve_hat