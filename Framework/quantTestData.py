#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import os
import numpy as np
import importlib
import scipy
 
import obq
importlib.reload(obq)
 
# =============================================================================
# SETTINGS
# =============================================================================
sInDir  = "TestBatches"
sOutDir = "TestBatches"   # gleicher Ordner -> überschreiben
 
vCaseFiles = [
    "REAL_FIXED_ONBIN_N2048_L213_20260505_130451",     # Dateinamen ohne .npz    
]
 
vMethods = ["SDQ","OBAQ"]              # gewünschte Methoden: ["SDQ", "OBAQ", ...]

for sCaseFile in vCaseFiles:

    sPath = os.path.join(sInDir, sCaseFile + ".npz")
    if not os.path.exists(sPath):
        raise FileNotFoundError(f"Case file not found: {sPath}")
    print(f"Loading: {sPath}")
    
    dictCase = dict(np.load(sPath))
    mx             = dictCase["mx"]      # (sN, sBatchSize)
    mr             = dictCase["mr"]
    mw             = dictCase["mw"]
    mrIdeal        = dictCase["mrIdeal"]
    mwIdeal        = dictCase["mwIdeal"]
    bMinPhase      = dictCase["bMinPhase"]
    sM             = dictCase["sM"]
    
    sN, sBatchSize = mx.shape
    
    ## Create Filters 
    ## ----- Triangular Matrix
    mOnes = np.ones((sN,sN))
    mSigDeltaFilt = np.tril(mOnes)


    for strMethod in vMethods:

        mb = np.zeros((sN, sBatchSize), dtype=float)
        
        for idxBatch in range(sBatchSize):

            vx = mx[:, idxBatch]
            vw = mw[:, idxBatch]

            if strMethod == "SDQ":
                vb, _, _ = obq.iterSequQ(vx,mSigDeltaFilt,0)

            elif strMethod == "OBAQ":
                if bMinPhase == True:
                    vw = scipy.signal.minimum_phase(vw, method='homomorphic')
                    vb, _, _ = obq.iterBlockQ(vx, vw, sM, 'grb') 
                else:
                    vb, _, _ = obq.iterBlockQ_OA(vx, vw, sM, 'grb')
                    
            elif strMethod == "oPWM":
                if bMinPhase == True:
                    vw = scipy.signal.minimum_phase(vw, method='homomorphic')
                    vb, _, _ = obq.iterBlockQ(vx, vw, sM, 'grb') 
                else:
                    vb, _, _ = obq.iterBlockQ_OA(vx, vw, sM, 'grb') 

            else:
                raise ValueError(f"Unknown quantization method: '{strMethod}'")

            mb[:, idxBatch] = vb

        dictCase[f"mb_{strMethod}"] = mb
        print(f"  [{strMethod}] done -> key: mb_{strMethod}")

    np.savez(os.path.join(sOutDir, sCaseFile + ".npz"), **dictCase)
    print(f"Saved: {sPath}\n")

print("Done.")