import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.signal import get_window

import sys
sys.path.append('../../01_Library')
# individual packages
import sg, sa, sp, obq, filt


def OptDFT(vRIX_t_L, mRIFw_M, vxM, vbM_init):
    """
    Args:
        vx: Input vector.
        vW: Desired spectral 
        K: K values for DFT
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    
    sM = len(vbM_init)
    sK = len(vRIX_t_L)
    
    mRIFw_M2 = mRIFw_M.T @ mRIFw_M                         # shape: (sM, sM)
    
    if np.all(vbM_init == 0):
        sMean = np.mean(vxM)
        bInit = np.full_like(vxM, 0)
        bInit[vxM >= sMean] = 1
    else:
        bInit = vbM_init.copy()
        
    vRIE    = np.zeros((sK,1)).flatten()
    vb_out  = np.zeros((sM,1)).flatten()
    # GUROBI
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    
    model = gp.Model("MIQP")
    model.setParam("OutputFlag", 0)     # 0 to Suppress Gurobi output
    model.setParam("TimeLimit",2)
    model.setParam("VarBranch", 3)
    model.setParam("MIPFocus", 3)       # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.9)   # Increase heuristic efforts
    model.setParam("Presolve", 2)       # More aggressive presolve
    model.setParam('Method', -1)
    
    with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
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
    
    sBlockErr = mRIFw_M @ (vxM - vb_out)
    sBlockErr = sBlockErr.T @ sBlockErr   
  
    # Output the solution
    if model.status == GRB.OPTIMAL:
        outTxt = f"Optimal solution found. ({model.SolCount})"
    else:
        outTxt = f"No Optimal solution found. ({model.SolCount})"
        
    return vb_out, sBlockErr, outTxt