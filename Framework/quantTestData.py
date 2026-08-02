#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.signal

import obq

# =============================================================================
# SETTINGS
# =============================================================================
sInDir  = "TestBatches"
sOutDir = "QuantBatches"

vCaseFiles = [
    "REAL_FIXED_ONBIN_20260802_075218_738557",     # file names without .npz
]

vMethods = ["SDQ","OBAQ"]              # ["SDQ", "OBAQ", "oPWM"]

os.makedirs(sOutDir, exist_ok=True)

for sCaseFile in vCaseFiles:

    sPath = os.path.join(sInDir, sCaseFile + ".npz")
    if not os.path.exists(sPath):
        raise FileNotFoundError(f"Case file not found: {sPath}")
    print(f"Loading: {sPath}")

    with np.load(sPath) as npzCase:
        mx        = npzCase["mx"]          # (sN, sBatchSize)
        vw        = npzCase["vw"]
        sM        = int(npzCase["sM"])
        bMinPhase = bool(npzCase["bMinPhase"])

    sN, sBatchSize = mx.shape

    ## Create Filters
    ## ----- Triangular Matrix
    mSigDeltaFilt = np.tril(np.ones((sN,sN)))

    if bMinPhase:
        vw = scipy.signal.minimum_phase(vw, method='homomorphic')

    dictQuant = {}

    for strMethod in vMethods:

        mb = np.zeros((sN, sBatchSize), dtype=float)

        for idxBatch in range(sBatchSize):

            vx = mx[:, idxBatch]

            if strMethod == "SDQ":
                vb, _, _ = obq.iterSequQ(vx, mSigDeltaFilt, 0)

            elif strMethod == "OBAQ":
                if bMinPhase:
                    vb, _, _ = obq.iterBlockQ(vx, vw, sM, 'grb')
                else:
                    vb, _, _ = obq.iterBlockQ_OA(vx, vw, sM, 'grb')

            elif strMethod == "oPWM":
                if bMinPhase:
                    vb, _, _ = obq.iterBlockQ(vx, vw, sM, 'grb')
                else:
                    vb, _, _ = obq.iterBlockQ_OA(vx, vw, sM, 'grb')

            else:
                raise ValueError(f"Unknown quantization method: '{strMethod}'")

            mb[:, idxBatch] = vb

        dictQuant[f"mb_{strMethod}"] = mb
        print(f"  [{strMethod}] done -> key: mb_{strMethod}")

    np.savez(os.path.join(sOutDir, sCaseFile + ".npz"), **dictQuant)
    print(f"Saved: {sCaseFile}\n")

print("Done.")