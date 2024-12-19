import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.signal import get_window

import sys
sys.path.append('../../01_Library')
# individual packages
import sg, sa, sp, obq, filt


def OptDFT(vx, vW, sK):
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
    sN = 2 ** int(np.ceil(np.log2(sLenVx)))
    vx = np.pad(vx, (0, sN - sLenVx), mode='constant')

    # Precompute sine and cosine coefficients for the DFT

    sLocMean = np.mean(vx)
    # Initialize the binary vector based on the mean
    bInit = np.full_like(vx, 0)
    bInit[sLocMean >= 0] = 1
    
    #identity matrix
    mLambda = np.diag(vW)
    mF      = sg.dftMat(sN, sK) 
    
    vFw_x   = mLambda @ (mF @ vx)
    mFw     = mLambda @ mF
    
    vRIFw_x = np.hstack((vFw_x.real, vFw_x.imag))
    mRIFw   = np.vstack((mFw.real, mFw.imag))
    
    ## Term 1 multiplication
    vRIFW_mRIFw = vRIFw_x.T @ mRIFw
    ## Term 2 multiplication
    mRIFw_RIFw  = mRIFw.T @ mRIFw
    
    vRIE    = np.zeros((len(vRIFw_x),1)).flatten()
    
    # GUROBI
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQCP")

    model.setParam("TimeLimit", 5)  # Increase numerical focus  
    #model.setParam("VarBranch", 3) 
    #model.setParam("MIPFocus", 0)  # Shift focus to finding good feasible solutions quickly
    #model.setParam("Heuristics", 0.9)  # Increase heuristic efforts
    #model.setParam("Presolve", 2)  # More aggressive presolve
    #model.setParam("Cuts", 3)  # More aggressive cut generation
    #model.setParam("MIPGap", 1e-12)
    model.setParam("TuneTimeLimit", 3600)
    
    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(sN, vtype=gp.GRB.BINARY, name="vb")
    
    for j in range(sN):
         vb[j].Start = bInit[j]
    
    model.update()

    # Objective function
    obj = gp.QuadExpr()
    
    # Linear term: -2 * vRIFW_mRIFw * (2 * vb - 1)
    for i in range(sN):
        obj.add(-2 * vRIFW_mRIFw[i] * (2 * vb[i] - 1)) 
    # Quadratic term: (2 * vb[i] - 1) * mRIFw_RIFw[i,j] * (2 * vb[j] - 1)
    for i in range(sN):
        for j in range(sN):
            if mRIFw_RIFw[i, j] != 0:  # Only include non-zero terms
                obj.add(mRIFw_RIFw[i, j] * (2 * vb[i] - 1) * (2 * vb[j] - 1))
    
    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.tune()

    # Output the solution
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found.")
        vb_out = np.array([2*vb[j].X - 1 for j in range(sN)])
    else:
        print("No optimal solution found.")
        vb_out = np.array([2*vb[j].X - 1 for j in range(sN)])
        

   
    return vb_out