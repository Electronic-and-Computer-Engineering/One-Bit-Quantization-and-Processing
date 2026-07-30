import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.signal import get_window

import sys
sys.path.append('../../01_Library')
# individual packages
import sg, sa, sp, obq, filt


def OptDFT(vEl_hat, mRIF_m, vx_m, vb_mInit):
    """
    Args:
        vx: Input vector.
        vW: Desired spectral 
        K: K values for DFT
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    
    sM = len(vx_m)
    
    mRIF_mM2 = mRIF_m.T @ mRIF_m                         # shape: (sM, sM)
    
    if np.all(vb_mInit == 0):
        sMean = np.mean(vx_m)
        bInit = np.full_like(vx_m, 0)
        bInit[vx_m >= sMean] = 1
    else:
        bInit = 2*vx_m.copy()+1
        
    vb_out  = np.zeros((sM,1)).flatten()
    # GUROBI
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    
    model = gp.Model("MIQP")
    model.setParam("OutputFlag", 0)     # 0 to Suppress Gurobi output
    model.setParam("MIPFocus", 2)               # Focus on proving optimality
    model.setParam("Method", 2)                 # Barrier method for root relaxation (robust for QCQP)
    model.setParam("Presolve", 2)               # Aggressive presolve
    #model.setParam("Heuristics", 0.75)          # Minimal heuristic distraction
    #model.setParam("Cuts", 3)                   # Most aggressive cutting planes
    #model.setParam("VarBranch", 1)              # Strong branching
    model.setParam("TimeLimit",2)
    # model.setParam("Cutpasses",10)
    #model.setParam("TuneTimeLimit",7200)
    
    term0 = vEl_hat.T @ vEl_hat
    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(sM, vtype=gp.GRB.BINARY, name="vb")
    
    for j in range(sM):
         vb[j].Start = bInit[j]
    
    model.update()
    
    vbDec = {j: 2 * vb[j] - 1 for j in range(sM)}
    vdHat_m = {j: vx_m[j] - vbDec[j] for j in range(sM)}

    # --- Term 1: -2 * X̃_L^T * (R * F_wM * b)
    term1 = gp.LinExpr()
    for i in range(len(vEl_hat)):
        for j in range(sM):           
            if mRIF_m[i, j] != 0:
                # Multiply with X̃_L[i]
                term1 += mRIF_m[i, j] * vEl_hat[i] * vdHat_m[j] 
    term1 *= 2  # Apply scalar factor as in the objective

    # --- Term 2: bᵀ * mRIFw_M2 * b
    term2 = gp.QuadExpr()
    for i in range(sM):
        for j in range(sM):
            if mRIF_mM2[i, j] != 0:
                term2 += vdHat_m[i] * mRIF_mM2[i, j] * vdHat_m[j]
    
    # --- Set the full objective function
    model.setObjective(term0 + term1 + term2, GRB.MINIMIZE)
    model.optimize()
    
    for i in range(sM):
            vb_out[i] = (2 * vb[i].X - 1)      
    
    sBlockErr = vEl_hat + mRIF_m @ (vx_m - vb_out)
    sBlockErr = sBlockErr.T @ sBlockErr   
  
    # Output the solution
    if model.status == GRB.OPTIMAL:
        outTxt = f"Optimal solution found. ({model.SolCount})"
    else:
        outTxt = f"No Optimal solution found. ({model.SolCount})"
        
    return vb_out, sBlockErr, outTxt