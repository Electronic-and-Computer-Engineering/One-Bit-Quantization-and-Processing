#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import obq, globalTools, filt

# =============================================================================
# SETTINGS
# =============================================================================
sInDir  = "TestBatches"
sOutDir = "QuantBatches"

vCaseFiles = [
    "REAL_FIXED_ONBIN_20260811_144745_922683",     # file names without .npz
]

vMethods = ["SDQ","OBBQ","OBBQ_lin"]              # ["SDQ", "OBAQ", "oPWM"]

os.makedirs(sOutDir, exist_ok=True)

for sCaseFile in vCaseFiles:

    sPath = os.path.join(sInDir, sCaseFile + ".npz")
    if not os.path.exists(sPath):
        raise FileNotFoundError(f"Case file not found: {sPath}")
    print(f"Loading: {sPath}")

    with np.load(sPath) as npzCase:
        mx        = npzCase["mx"]          # (sN, sBatchSize)
        vw        = npzCase["vw"]
        vwIdeal   = npzCase["vwIdeal"]
        sM        = int(npzCase["sM"])

    sN, sBatchSize = mx.shape

    ## Create Filters
    ## ----- Triangular Matrix
    mSigDeltaFilt = np.tril(np.ones((sN,sN)))
    vwMin,_,dInfo = filt.prunOptimal(vw, sW0Rel=0.1, sMetric='L2', bRequireMinPhase=True)

    dictQuant = {}

    for strMethod in vMethods:

        mb = np.zeros((sN, sBatchSize), dtype=float)       

        for idxBatch in range(sBatchSize):
            if idxBatch == 0:
                progressBlock = globalTools.SimpleProgressBar(sBatchSize, width=40, prefix = strMethod, fill="█", empty=" ", end=" ✓")                       

            vx = mx[:, idxBatch]

            if strMethod == "SDQ":
                with np.errstate(divide='ignore'):
                    vb, _, _ = obq.iterSequQ(vx, mSigDeltaFilt, 0)

            elif strMethod == "OBBQ":
                    vb, _, _ = obq.iterBlockQ(vx, vwMin, sM, 'grb', bSilent = True)
                    
            elif strMethod == "OBBQ_lin":
                    vb, _, _ = obq.iterBlockQ_OA(vx, vw, sM, 'grb', bSilent = True)

            else:
                raise ValueError(f"Unknown quantization method: '{strMethod}'")

            mb[:, idxBatch] = vb
            progressBlock.update(idxBatch+1)

        dictQuant[f"mb_{strMethod}"] = mb
        print(f"  [{strMethod}] done -> key: mb_{strMethod}")

    np.savez(os.path.join(sOutDir, sCaseFile + ".npz"), **dictQuant)
    print(f"Saved: {sCaseFile}\n")

print("Done.")