import numpy as np
import gurobipy as gp
import scipy.signal as sigP
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

    if np.all(vb_mInit == 0):
        sMean = np.mean(vx_m)
        bInit = (vx_m >= sMean).astype(int)
    else:
        bInit = ((vb_mInit + 1) // 2).astype(int)  # Map {-1,1} -> {0,1}
        
    vb_out = np.zeros(sM, dtype=float)
    # GUROBI
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQCP)
    
    model = gp.Model("MIQP")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", 0.5)  # Increase numerical focus  
    model.setParam("VarBranch", 3) 
    model.setParam("MIPFocus", 0)  # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.9)  # Increase heuristic efforts
    model.setParam("Presolve", 2)  # More aggressive presolve
    model.setParam("Cuts", 3)  # More aggressive cut generation
    model.setParam("MIPGap", 0)
    
    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(sM, vtype=gp.GRB.BINARY, name="vb")
    
    for j in range(sM):
         vb[j].Start = int(bInit[j])
    
    model.update()
    
    vbDec = {j: 2 * vb[j] - 1 for j in range(sM)}
    #vdHat_m = {j: vx_m[j] - vbDec[j] for j in range(sM)}

    # --- Term 1: -2 * X̃_L^T * (R * F_wM * b)
    obj = gp.QuadExpr()

    for i in range(mRIF_m.shape[0]):
        se = vEl_hat[i] + gp.quicksum(mRIF_m[i, j] * (vx_m[j] - vbDec[j]) for j in range(sM))     
        obj += se * se

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    for i in range(sM):
            vb_out[i] = (2 * vb[i].X - 1)      
    
    sBlockErr = vEl_hat + mRIF_m @ (vx_m - vb_out) 
  
    # Output the solution
    if model.status == GRB.OPTIMAL:
        outTxt = f"Optimal solution found. ({model.SolCount})"
        bOpt = True
    else:
        outTxt = f"No Optimal solution found. ({model.SolCount})"
        bOpt = False
        
    return vb_out, sBlockErr, outTxt, bOpt