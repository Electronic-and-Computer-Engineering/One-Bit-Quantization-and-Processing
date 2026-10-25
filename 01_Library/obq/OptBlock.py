import numpy as np
import gurobipy as gp
from gurobipy import GRB

def OptBlock(vx, mW, vCe):
    """
    Args:
        vx:         Input vector.
        mW:         Weight matrix.
        vCe:        Convolutional sum of previous found solutions
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
        outTxt: Status text
        sTgrb: Gurobi's own solve time in seconds, without the model build
        bLimit: True if the solver stopped at its time limit, i.e. the returned
                solution is the incumbent and not the proven optimum
    """
    nVars = mW.shape[1]
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQP")
    model.setParam("OutputFlag", 0)     # 0 to Suppress Gurobi output
    model.setParam("MIPGap",1e-5)
    model.setParam("TimeLimit", 2)
    model.setParam("VarBranch", 3)

    #sOptOff = (mW.shape[0] - len(vx) + 1) // 2
    #Initialization
    bInit = (vx >= 0).astype(int)

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
    sTgrb  = model.Runtime                          # solve time, model build excluded
    bLimit = (model.status == GRB.TIME_LIMIT)       # incumbent, not a proven optimum
    if model.status == GRB.OPTIMAL:
        outTxt = f"Optimal solution found. ({model.SolCount}), t = {sTgrb:.4f}s"
    elif bLimit:
        outTxt = f"Time limit, gap = {model.MIPGap:.2e} ({model.SolCount}), t = {sTgrb:.4f}s"
    else:
        outTxt = f"No Optimal solution found. ({model.SolCount}), t = {sTgrb:.4f}s"

    vb_out = np.array([2 * vb[j].X - 1 for j in range(nVars)])  # Extract optimized solution
    ve = mW @ (vx - vb_out) + vCe  # Updated error considering the optimization
    
    return vb_out, ve, outTxt, sTgrb, bLimit