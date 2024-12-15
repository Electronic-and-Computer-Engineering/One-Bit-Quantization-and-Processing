#sympi
import sympy as sypy
import scipy.linalg as scLinAlg
import numpy as np

import sg,misc

def iterBlockQ(vxSym, vwSym, vx, vw, sM, strFileNameTxt):
    """
    Args:
        vx: Input vector (SymPy Matrix).
        vw: Weight vector (SymPy Matrix).
        sM: Number of decisions (block size).
        
    Returns:
        vb: Quantized one-bit vector (SymPy Matrix)
        ve: Error vector (SymPy Matrix)
    """
    swLen = len(vwSym)
    sxLen = len(vxSym)
    
    veSym = sypy.zeros(sxLen, 1)  # Error vector
    vbSym = sypy.zeros(sxLen, 1)  # Quantized vector
    ve = np.zeros((sxLen,1)).flatten()         
    vb = np.zeros((sxLen,1)).flatten()   

    vwFull = np.zeros((sxLen,1)).flatten()
    vwFull[0:swLen] = vw

    vwFullSym = sypy.zeros(sxLen, 1)
    for i in range(swLen):
        vwFullSym[i] = vwSym[i] # Copy vw into the beginning of vwFull

    ve_init = np.zeros((sM,1)).flatten()
    vbBlock = np.zeros((sM,1)).flatten() 
    veBlock = np.zeros((sM,1)).flatten() 

    if sxLen % sM != 0:
        print("vx should be a multiple of sM")
        return None, None

    # Prepare the initial weight matrix using Toeplitz structure
    txt = f"============= W_0:\n"
    misc.printToTxt(txt, strFileNameTxt, True)   
    mW_0 = sypy.Matrix(np.tril(scLinAlg.toeplitz([vwFullSym[i, 0] for i in range(sM)], [vwFullSym[i, 0] for i in range(sM)])))
    ltxMW_0= sypy.latex(mW_0, mat_delim='', mat_str='bmatrix')
    txt = f"W^{(0)} = {ltxMW_0}\n\n"
    misc.printToTxt(txt, strFileNameTxt, True)

    for m in range(sxLen // sM):
        ve_hat = ve_init.copy()
        txt = f"============= {m}:\n"
        ve_hatSym = ""        
        misc.printToTxt(txt, strFileNameTxt, True)   
        for k in range(m):
            sRowIdx = sM * (m - k)
            sColIdx = (m - k - 1) * sM
            mW_m = sypy.Matrix(scLinAlg.toeplitz(vwFullSym[sRowIdx:sRowIdx+sM],np.flip(vwFullSym[sColIdx:sColIdx+sM])))
            ltxMW_k= sypy.latex(mW_m, mat_delim='', mat_str='bmatrix')
            txt = f"W^{(m - k)} = {ltxMW_k}\n"
            
            misc.printToTxt(txt, strFileNameTxt, True)
            
            ve_hat += mW_m * (vx[sM*k:sM*(k+1), 0] - vb[sM*k:sM*(k+1), 0])
       
    mCom = sg.binaryComb(sM)       
     
        
    return vbSym