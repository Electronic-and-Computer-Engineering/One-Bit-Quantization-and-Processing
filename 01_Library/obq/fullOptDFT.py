import numpy as np
import gurobipy as gp
from gurobipy import GRB

import sys
sys.path.append('../../01_Library')
# individual packages
import sg, sa, sp, obq, filt


def fullOptDFT(vx, vW, sK):
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
    
    vRIE    = np.zeros((len(vRIFw_x),1)).flatten()
    
    #vx_ps = np.linalg.inv(vRIFw_x.T @ vRIFw_x) @ mRIFw.T @ vRIFw_x
    # GUROBI
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQCP")

    model.setParam("TimeLimit", 20)  # Increase numerical focus
    model.setParam("VarBranch", 3) 
    model.setParam("MIPFocus", 3)  # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.3)  # Increase heuristic efforts
    model.setParam("Presolve", 2)  # More aggressive presolve
    #model.setParam("Cuts", 3)  # More aggressive cut generation
    model.setParam("MIPGap", 1e-05)
    #model.setParam("TuneTimeLimit", 2400)
    
    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(sN, vtype=gp.GRB.BINARY, name="vb")
    
    #for j in range(sN):
    #     vb[j].Start = bInit[j]
    
    model.update()

    # Objective function
    obj = gp.QuadExpr()
    
    for i in range(mRIFw.shape[0]):  # Rows of vRIE
        se = 0
        se = gp.quicksum(
            mRIFw[i,j]*(2*vb[j] - 1) for j in range(mRIFw.shape[1])
            )
        
        obj += (vRIFw_x[i] - se) * (vRIFw_x[i] - se)
    
    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)
    
    model.optimize()

    # Output the solution
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found.")
        vb_out = np.array([2*vb[j].X - 1 for j in range(sN)])
    else:
        print("No optimal solution found.")
        vb_out = np.array([2*vb[j].X - 1 for j in range(sN)])
        

   
    return vb_out