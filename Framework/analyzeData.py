#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import sa, fStg, filt


# =============================================================================
# SETTINGS
# =============================================================================
sSigDir   = "TestBatches"
sQuantDir = "QuantBatches"
sEvalDir  = "EvalBatches"

vCaseFiles = [
    "REAL_FIXED_ONBIN_20260811_144745_922683",     # file names without .npz
]

bPlot = True                       # False -> compute and save only

os.makedirs(sEvalDir, exist_ok=True)

# =============================================================================
# EVALUATION
# =============================================================================
for sCaseFile in vCaseFiles:
    # ----- load -------------------------------------------------------------
    sPathSig   = os.path.join(sSigDir,   sCaseFile + ".npz")
    sPathQuant = os.path.join(sQuantDir, sCaseFile + ".npz")

    with np.load(sPathSig) as npzSig:
        mx      = npzSig["mx"]             # (sN, sBatchSize)
        vw      = npzSig["vw"]             # shaping FIR (Kaiser)
        vr      = npzSig["vr"]             # reconstruction FIR (Kaiser)
        vrIdeal = npzSig["vrIdeal"]        # reconstruction FIR (ideal bin mask)

    with np.load(sPathQuant) as npzQuant:
        dictMb = {k[3:]: npzQuant[k] for k in npzQuant.files if k.startswith("mb_")}

    sN, sBatchSize = mx.shape

    # ----- error sources: reference (zero error) plus every method ----------
    dictErr = {"REF": np.zeros_like(mx)}
    for strMethod in sorted(dictMb):
        dictErr[strMethod] = mx - dictMb[strMethod]

    # ----- metrics ----------------------------------------------------------
    vRIdealFFT = np.fft.fft(vrIdeal)

    # columns: MSE, SNR, PSNR
    dictRes = {}

    for strMethod, mErr in dictErr.items():

        mResIdeal = np.zeros((sBatchSize, 3))
        mResReal  = np.zeros((sBatchSize, 3))

        for idxBatch in range(sBatchSize):

            vx   = mx[:, idxBatch]
            vErr = mErr[:, idxBatch]

            mResIdeal[idxBatch, :] = sa.evalN(filt.reconIdeal(vErr, vRIdealFFT),
                                              filt.reconIdeal(vx,   vRIdealFFT))

            mResReal[idxBatch, :]  = sa.evalN(filt.reconReal(vErr, vr),
                                              filt.reconReal(vx,   vr))

        dictRes[f"ideal_{strMethod}"] = mResIdeal
        dictRes[f"real_{strMethod}"]  = mResReal

    # ----- save npz ---------------------------------------------------------
    np.savez(os.path.join(sEvalDir, sCaseFile + "_eval.npz"), **dictRes)

    # ----- save csv and markdown --------------------------------------------
    vKeys = sorted(dictRes)

    fStg.writeEvalCsv(os.path.join(sEvalDir, sCaseFile + "_eval.csv"),
                 dictRes, vKeys, sBatchSize)

    fStg.writeEvalMarkdown(os.path.join(sEvalDir, sCaseFile + "_eval.md"),
                      sCaseFile, dictRes, vKeys,
                      os.path.join(sSigDir, sCaseFile + ".md"))

    # ----- plot -------------------------------------------------------------
    if bPlot:
        fStg.plotEval(mx, dictMb, dictRes, vw, sCaseFile)