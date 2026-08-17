import numpy as np
import gurobipy as gp
from gurobipy import GRB


def OptDFT(vx_m, vRIX_t_L, mRIFw_m, sM, fTune=False):
    """
    Solves the per-block Binary Quadratic Program (BQP_p).

    Args:
        vx_m:      Current M-block signal samples,        shape: (sM,)
        vRIX_t_L:  Spectral target E^l + Fw^M @ x^M,     shape: (2K,)
        mRIFw_m:   Real/imag stacked projection matrix,   shape: (2K x sM)
        sM:        Block length
        fTune:     If True, runs Gurobi's parameter tuner (model.tune())
                   on THIS block instead of solving with fixed settings.
                   Use this by picking one call site in your loop (e.g.
                   via `if m == some_index: fTune=True`) to tune on a
                   real, representative block.
    Returns:
        vb_out:    Quantized one-bit block,               shape: (sM,)
        sBlockErr: Scalar block error
        tune_info: ONLY returned if fTune=True (3-tuple instead of 2).
                   dict with:
                     n_found       -- wie viele Parameter-Sets gefunden wurden
                     prm_files     -- Liste der gespeicherten .prm-Dateien,
                                       bestes zuerst (rank0..rank4)
                     best_runtime  -- Laufzeit mit dem besten gefundenen Set
                     best_objval   -- damit erreichter Zielfunktionswert
    """
    vb_ls  = np.linalg.lstsq(mRIFw_m, vRIX_t_L, rcond=None)[0]
    vbInit = np.where(vb_ls >= 0, 1.0, 0.0)

    model = gp.Model("BQP")

    if fTune:
        model.setParam("OutputFlag", 1)
        model.setParam("TuneTimeLimit", 1800)
        model.setParam("TimeLimit",    2)
    else:
        model.setParam("OutputFlag",   0)
        model.setParam("TimeLimit",    1)
        # model.setParam("VarBranch",   -1)
        # model.setParam("MIPFocus",     2)
        # model.setParam("Heuristics",   0.5)
        # model.setParam("Presolve",     2)
        # model.setParam("Cuts",         1)
        # model.setParam("MIPGap",       0)
        # model.setParam("NumericFocus", 1)

    vb = model.addVars(sM, vtype=GRB.BINARY, name="vb")
    for j in range(sM):
        vb[j].Start = int(vbInit[j])
    model.update()

    vbDec = {j: 2 * vb[j] - 1 for j in range(sM)}

    obj = gp.QuadExpr()
    for i in range(mRIFw_m.shape[0]):
        se = vRIX_t_L[i] - gp.quicksum(
            mRIFw_m[i, j] * vbDec[j] for j in range(sM)
        )
        obj += se * se

    model.setObjective(obj, GRB.MINIMIZE)

    if fTune:
        model.setParam("TuneTrials", 3)     # Wiederholungen pro Kandidat
        model.setParam("TuneOutput", 1)     # etwas Fortschritts-Log
        model.tune()

        # WICHTIG: model.tune() loest das Modell NICHT -- es durchsucht nur
        # den Parameterraum und laedt KEINE Loesung ins Modell. Ohne
        # getTuneResult()+optimize() waere vb[j].X weiter unten nicht
        # verfuegbar und wuerde crashen.
        n_found = model.tuneResultCount
        prm_files = []
        for i in range(min(n_found, 5)):
            model.getTuneResult(i)
            fname = f"tune_rank{i}.prm"
            model.write(fname)
            prm_files.append(fname)

        # bestes gefundenes Set laden und einmal wirklich loesen
        model.getTuneResult(0)
        model.optimize()

        tune_info = {
            "n_found":     n_found,
            "prm_files":   prm_files,          # beste zuerst, rank0..rank4
            "best_runtime": model.Runtime,     # Laufzeit mit bestem Set
            "best_objval":  model.ObjVal,      # erreichter Zielfunktionswert
        }
    else:
        model.optimize()

    vb_out = np.array([2 * vb[j].X - 1 for j in range(sM)])
    sBlockErr = float(np.linalg.norm(vRIX_t_L - mRIFw_m @ vb_out) ** 2)

    bOptimal = model.status == GRB.OPTIMAL
    print(" ✓ Optimal   " if bOptimal else " ✗ SubOptimal", end='', flush=True)

    if fTune:
        return vb_out, sBlockErr, tune_info
    return vb_out, sBlockErr