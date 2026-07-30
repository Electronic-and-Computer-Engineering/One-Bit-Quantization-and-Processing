import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.signal import get_window

import sys
sys.path.append('../../01_Library')
# individual packages
import sg, sa, sp, obq, filt


def OptDFT(vx, vW, sK, vbL, sM):
    """
    Args:
        vx: Input vector.
        vW: Desired spectral 
        K: K values for DFT
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    # Zero-Pad to the next pow2 value
    sLenVx = len(vx)
    #sN     = 2 ** int(np.ceil(np.log2(sLenVx)))
    #vx     = np.pad(vx, (0, sN - sLenVx), mode='constant')
    # Precompute sine and cosine coefficients for the DFT
    
    sL = len(vbL)
    vxOptBlock = vx[sL::]
    sBlockMean = np.mean(vxOptBlock)
    # Initialize the binary vector based on the mean
    bInit = np.full_like(vxOptBlock, 0)
    bInit[sBlockMean >= 0] = 1
    
    # Windowing
    #vWm = get_window("blackman", sM)
    #vx[sL::] = vxOptBlock * vWm
    #vx = vx * w
    
    # DFT matrix (K x N)
    F = sg.dftMat(sLenVx, sK)                 # shape: (sK x sBlockLen)
    Fw = np.diag(vW) @ F                      # shape: (sK x sBlockLen)

    # Split DFT matrix
    Fw_L = Fw[:, :sL]                         # shape: (sK x sL)
    Fw_M = Fw[:, sL:]                         # shape: (sK x sM)

    # Spectral target with weighting
    vX_t = Fw @ vx                            # shape: (sK,)
    vX_t_L = vX_t - Fw_L @ vbL                # shape: (sK,)

    # Real/Imaginary split for real-valued optimization
    vRIX_t_L = np.hstack([vX_t_L.real, vX_t_L.imag])       # shape: (2K,)
    mRIFw_M  = np.vstack([Fw_M.real, Fw_M.imag])           # shape: (2K, sM)
    mRIFw_M2 = mRIFw_M.T @ mRIFw_M                         # shape: (sM, sM)
    
    vRIE    = np.zeros((len(vRIX_t_L),1)).flatten()
    vb_out  = np.zeros((sM,1)).flatten()
    # GUROBI
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQCP")
    
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 0.5)  # Increase numerical focus  
    model.setParam("VarBranch", 3) 
    model.setParam("MIPFocus", 0)  # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.9)  # Increase heuristic efforts
    model.setParam("Presolve", 2)  # More aggressive presolve
    model.setParam("Cuts", 3)  # More aggressive cut generation
    model.setParam("MIPGap", 0)
    #model.setParam("TuneTimeLimit", 3600)
    
    term0 = vRIX_t_L.T @ vRIX_t_L
    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(sM, vtype=gp.GRB.BINARY, name="vb")
    
    for j in range(sM):
         vb[j].Start = bInit[j]
    
    model.update()

    # --- Term 1: -2 * X̃_L^T * (R * F_wM * b)
    term1 = gp.LinExpr()
    for i in range(len(vRIX_t_L)):
        # Inner product: sum_j (R * F_wM)[i, j] * b[j]
        for j in range(sM):
            vbDec = 2*vb[j] - 1
            if mRIFw_M[i, j] != 0:
                # Multiply with X̃_L[i]
                term1 += mRIFw_M[i, j] * vRIX_t_L[i] * vbDec 
    term1 *= -2  # Apply scalar factor as in the objective

    # --- Term 2: bᵀ * mRIFw_M2 * b
    term2 = gp.QuadExpr()
    for i in range(sM):
        vbDec_i = 2*vb[i] - 1
        for j in range(sM):
            vbDec_j = 2*vb[j] - 1
            
            if mRIFw_M2[i, j] != 0:
                term2 += vbDec_i * mRIFw_M2[i, j] * vbDec_j
    
    # --- Set the full objective function
    model.setObjective(term0 + term1 + term2, GRB.MINIMIZE)
    model.optimize()
    
    for i in range(sM):
            vb_out[i] = (2 * vb[i].X - 1)      
    
    blockErr = term0 + -2 * vRIX_t_L.T @ (mRIFw_M @ vb_out) + vb_out.T @ (mRIFw_M2 @ vb_out)         
    # Output the solution
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found.")
    else:
        print("No optimal solution found.")
        
    return vb_out, blockErr