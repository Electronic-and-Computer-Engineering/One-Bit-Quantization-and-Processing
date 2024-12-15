import numpy as np
import gurobipy as gp
from gurobipy import GRB

def fullOptN(vx, mW, vBStart):
    """
    Args:
        vx: Input vector.
        mW: Weight matrix.
        vC: Constant/Init vector.
        sL: Number of Decisions
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    nVars = mW.shape[1]
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
        #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQP")

    model.setParam("TimeLimit", 20)  # Increase numerical focus
    model.setParam("VarBranch", 3) 
    model.setParam("MIPFocus", 3)  # Shift focus to finding good feasible solutions quickly
    model.setParam("Heuristics", 0.3)  # Increase heuristic efforts
    #model.setParam("Presolve", 2)  # More aggressive presolve
    #model.setParam("Cuts", 3)  # More aggressive cut generation
    model.setParam("MIPGap", 1e-9)
    #model.setParam("TuneTimeLimit", 2400)

    # Decision variables (vb) as binary, mapped to {-1, 1} in the objective
    vb = model.addVars(nVars, vtype=GRB.BINARY, name="vb")
    # for j in range(nVars):
    #     vb[j].Start = vBStart[j]

    model.update()

    # Objective function
    obj = gp.QuadExpr()
    for i in range(mW.shape[0]):  # Rows of mW
        se = 0
        for j in range(nVars):  # Elements of vb and vx
            # Adjust the vb[j] from {0, 1} to {-1, 1}
            vbDec = 2*vb[j] - 1
            sd = vx[j] - vbDec
            # Contribution of each element to the quadratic term
            se += mW[i, j] * sd
        #obj += se * se
        obj.add(se * se)

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    # Output the solution
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        vb_out = np.array([2 * vb[j].X - 1 for j in range(nVars)])
        ve = mW @ vx - mW @ vb_out  # Ensure dimensions align   
    else:
        print("No optimal solution found.")
        vb_out = np.array([2 * vb[j].X - 1 for j in range(nVars)])
        ve = mW @ vx - mW @ vb_out  # Ensure dimensions align   
   
    return vb_out, ve