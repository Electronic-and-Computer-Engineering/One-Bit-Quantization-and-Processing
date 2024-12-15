import numpy as np
import gurobipy as gp
from gurobipy import GRB

def OptBlock(vx, mW, vCe):
    """
    Args:
        vx:         Input vector.
        mW:         Weight matrix.
        vE_hat:     Previous convolutional Error.
        vBStart:    Initial Decision 
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    nVars = mW.shape[1]
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQP")
    model.setParam("OutputFlag", 0)     # 0 to Suppress Gurobi output
    model.setParam("TimeLimit",10)
    model.setParam("VarBranch", 3)
    model.setParam("MIPFocus", 3)       # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.9)   # Increase heuristic efforts
    model.setParam("Presolve", 2)       # More aggressive presolve
    model.setParam('Method', -1)

    #sOptOff = (mW.shape[0] - len(vx) + 1) // 2
    #Initialization
    sLocMean = np.mean(vx)
    # Initialize the binary vector based on the mean
    bInit = np.full_like(vx, 0)
    bInit[sLocMean >= 0] = 1

    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(nVars, vtype=GRB.BINARY, name="vb")
    for j in range(nVars):
         vb[j].Start = bInit[j]

    model.update()

    # Objective function
    obj = gp.QuadExpr()

    for i in range(mW.shape[0]):  # Rows of mW
        se = vCe[i].copy() 

        for j in range(nVars):  # Elements of vb and vx
            # Adjust the vb[j] from {0, 1} to {-1, 1}
            vbDec = 2*vb[j] - 1
            sd = vx[j] - vbDec
            # Contribution of each element to the quadratic term
            se = se + mW[i, j] * sd 
        obj += se * se 

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Output the solution
    if model.status == GRB.OPTIMAL:
        print(f"Optimal solution found. ({model.SolCount})", end=" ")
    else:
        print(f"No Optimal solution found. ({model.SolCount})", end=" ")

    vb_out = np.array([2 * vb[j].X - 1 for j in range(nVars)])  # Extract optimized solution
    ve = mW @ (vx - vb_out) + vCe  # Updated error considering the optimization
    
    return vb_out, ve