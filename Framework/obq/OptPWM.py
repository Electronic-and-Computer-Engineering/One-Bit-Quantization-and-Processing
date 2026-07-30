import numpy as np
import gurobipy as gp
from gurobipy import GRB

def OptPWM(vx, vw, sP, sKappap, fix_Mp=True, fix_kappa=True):

    sN = len(vx)
    sL = len(vw)

    m = gp.Model("MIQP")
    m.setParam("OutputFlag", 1)
    m.setParam("TimeLimit", 240)
    m.setParam("MIPFocus", 2)
    m.setParam("Heuristics", 0.75)
    m.setParam("Presolve", 2)
    m.setParam("Cuts", 2)
    m.setParam("Method", -1)
    m.setParam("Threads", 0)
    m.setParam("MIPGap", 0.01)

    print("N =", sN, "P =", sP, "L =", sL)
    print("vx finite:", np.isfinite(vx).all())
    print("vw finite:", np.isfinite(vw).all())
    
    # --------------------------------------------------
    # 2) Zone lengths Mp
    # --------------------------------------------------
    if fix_Mp:
        if sN % sP != 0:
            raise ValueError("N must be divisible by P when fix_Mp=True.")
        sMp = sN // sP
        vMp = {p: sMp for p in range(sP)}
    else:
        print("Case fix_Mp = False is not implemented yet.")
        return None
        #vMp = m.addVars(sP, vtype=GRB.INTEGER, lb=1, ub=sN, name="Mp")
        #m.addConstr(gp.quicksum(vMp[p] for p in range(sP)) == sN, name="partition")
    
    # --------------------------------------------------
    # 3) Global zone offsets Mhat_p
    # --------------------------------------------------
    if fix_Mp:
        vMhat = {p: p * sMp for p in range(sP)}
    else:
        print("Case fix_Mp = False is not implemented yet.")
        return None    
    # --------------------------------------------------
    # 4) Define kappa
    # --------------------------------------------------    
    if fix_kappa:
        vkappa = {p: int(sKappap) for p in range(sP)}
    else:
        vkappa = m.addVars(sP, vtype=GRB.INTEGER, lb=0, ub=sMp - 1, name="vkappa")
    # --------------------------------------------------
    # 5) HIGH Zones
    # --------------------------------------------------    
    vNon = m.addVars(sP, vtype=GRB.INTEGER, lb=0, ub=sMp, name="vNon")  
    
    for p in range(sP):
        m.addConstr(vNon[p] + vkappa[p] <= vMp[p], name=f"non_bound_{p}")
    # -------------------------------------------------------
    # 6) Define binary HIGH/LOW state h[n]
    #    vHigh[n] = 1  -> HIGH
    #    vHigh[n] = 0  -> LOW
    # -------------------------------------------------------
    vHigh = m.addVars(sN, vtype=GRB.BINARY, name="vHigh")
    
    if fix_kappa:
        for p in range(sP):
            start = vMhat[p]
            for n in range(start, start + vMp[p]):
                n_loc = n - start
                
                if n_loc < vkappa[p]:
                    m.addConstr(vHigh[n] == 0,name=f"low_before_block_n{n}")
                else:
                    rhs = n_loc - vkappa[p] + 1
                    m.addGenConstrIndicator(vHigh[n], True, vNon[p] >= rhs,name=f"force_high_{n}")
                    m.addGenConstrIndicator(vHigh[n], False, vNon[p] <= rhs - 1,name=f"force_low_{n}")
    else:
   
        # vkappa[p] variable:
        # vHigh[n] = 1  <=>  n in [vkappa[p], vkappa[p]+vNon[p])
        # vBefore[n]: 1 = before block, 0 = after block
    
        vBefore = m.addVars(sN, vtype=GRB.BINARY, name="vBefore")
    
        for p in range(sP):
            start = vMhat[p]   
            for n in range(start, start + vMp[p]):    
                n_loc = n - start    
                # HIGH: enforce n inside block
                m.addConstr(vkappa[p] <= n_loc + (vMp[p] + 1) * (1 - vHigh[n]), name=f"high_left_{n}")   
                m.addConstr(vkappa[p] + vNon[p] >= n_loc + 1 - (vMp[p] + 1) * (1 - vHigh[n]), name=f"high_right_{n}")
    
                # LOW: enforce n outside block
                # before block
                m.addConstr(vkappa[p] >= n_loc + 1 - (vMp[p] + 1) * (vHigh[n] + (1 - vBefore[n])), name=f"low_before_{n}")    
                # after block
                m.addConstr(vkappa[p] + vNon[p] <= n_loc + (vMp[p] + 1) * (vHigh[n] + vBefore[n]), name=f"low_after_{n}")
    
    # --------------------------------------------------
    # 7) Convert binary HIGH signal to bipolar signal b[n]
    # --------------------------------------------------

    vb = m.addVars(sN, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="vb")
    for n in range(sN):
        m.addConstr(vb[n] == 2 * vHigh[n] - 1, name=f"b_def_{n}")
    
    # Objective function
    obj = gp.QuadExpr()

    for n in range(sN):
        e_n = gp.quicksum(vw[k] * (vx[n-k] - vb[n-k])
                          for k in range(sL)
                          if n - k >= 0)
        obj += e_n * e_n
                    
    m.setObjective(obj, GRB.MINIMIZE)     
    m.optimize()

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    if m.status == GRB.OPTIMAL:
        print(f"Optimal solution found. ({m.SolCount})")
    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        return None
    else:
        print(f"No Optimal solution found. ({m.SolCount})")
    
    vb_out = np.array([vb[j].X for j in range(sN)])
    return vb_out
    
