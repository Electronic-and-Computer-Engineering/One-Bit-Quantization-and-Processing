import gurobipy as gp
from gurobipy import GRB

def OptBlock_gram(vx, mW, vCe):

    """
    Block subproblem in Gram form -- same argmin as building the objective
    row by row, but the model has sM^2 terms instead of sK*sM^2.
 
    With b = 2u-1, u binary, and d = vx - b = a - 2u, a = vx + 1:
 
        || mW d + vCe ||^2  =  4 u^T G u  -  4 (G a + g)^T u  +  const
        G = mW^T mW  (sM x sM),   g = mW^T vCe,
        const = a^T G a + 2 g^T a + vCe^T vCe
 
    G is PSD and constant over all blocks, so pass it in precomputed. Only g
    changes per block. The diagonal is deliberately NOT folded into the linear
    term: that would destroy positive semidefiniteness and force Gurobi into
    non-convex mode.
 
    Input-Arguments:
        vx:         Input vector (length sM).
        mW:         Filter matrix (sK x sM).
        vCe:        Convolutional sum of previous found solutions (length sK).
        mG:         Precomputed mW^T mW. Built here when None.
 
    Returning:
        vb:     Quantized one-bit vector (+-1).
        ve:     Error vector, mW (vx - vb) + vCe  (length sK).
        outTxt: Solver status for the progress bar.
    """
    nVars = mW.shape[1]
    mG = mW.T @ mW
    vg = mW.T @ vCe
    va = vx + 1.0

    mQ = 4.0 * mG
    vc = -4.0 * (mG @ va + vg)
    sConst = va @ mG @ va + 2.0 * vg @ va + vCe @ vCe
    #Mixed-Integer Quadratically Constrained Quadratic Programming (MIQP)
    model = gp.Model("MIQP")
    model.setParam("OutputFlag", 0)     # 0 to Suppress Gurobi output
    model.setParam("MIPGap",1e-3)
    model.setParam("TimeLimit", 2)
    model.setParam("VarBranch", 3)

    # Decision variables as binary, mapped to {-1, 1} via b = 2u - 1
    u = model.addMVar(nVars, vtype=GRB.BINARY, name="u")
    u.Start = (vx >= 0).astype(int)
 
    model.setObjective(u @ mQ @ u + vc @ u + sConst, GRB.MINIMIZE)
    model.optimize()
 
    if model.status == GRB.OPTIMAL:
        outTxt = f"Optimal solution found. ({model.SolCount})"
    else:
        outTxt = f"No Optimal solution found. ({model.SolCount})"
 
    vb = 2.0 * u.X - 1.0
    ve = mW @ (vx - vb) + vCe
 
    return vb, ve, outTxt