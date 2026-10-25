#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import time
sys.path.append('../01_Library')

import sg, obq, globalTools, filt

# =============================================================================
# SETTINGS
# =============================================================================
sInDir  = "TestBatches"
sOutDir = "QuantBatches"

vCaseFiles = [
    "REAL_FIXED_ONBIN_20261025_121302_661850", #"REAL_FIXED_ONBIN_20260817_181320_775506",     # file names without .npz
]

vMethods = ["SDQ","OBBQ_bf16"]              # ["SDQ", "OBAQ", "oPWM"]

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
    vwMin,_,dInfo = filt.prunOptimal(vw, sW0Rel=0.1, sMetric='L2', bRequireMinPhase=True)
    mSigDeltaFilt = np.tril(np.ones((sN,sN)))
    mSignFilt     = np.identity(sN)
    mfiltMatrix   = sg.lowerToeplitz(vwMin,sN,bSparse=False)


    ## Load already computed results, so other methods need NOT be recomputed
    sOutPath = os.path.join(sOutDir, sCaseFile + ".npz")
    dictQuant = {}
    if os.path.exists(sOutPath):
        with np.load(sOutPath) as npzOld:
            dictQuant = {k: npzOld[k] for k in npzOld.files}
        print(f"  Existing results: {sorted(k[3:] for k in dictQuant if k.startswith('mb_'))}")

    for strMethod in vMethods:

        mb = np.zeros((sN, sBatchSize), dtype=float)       
        vTs    = np.zeros(sBatchSize)   # signal time, methods without blocks
        lstTb  = []                     # block times, methods with blocks
        lstLim = []
        
        for idxBatch in range(sBatchSize):
            if idxBatch == 0:
                progressBlock = globalTools.SimpleProgressBar(sBatchSize, width=40, prefix = strMethod, fill="█", empty=" ", end=" ✓")                       

            vx = mx[:, idxBatch]
            vTb, vLim, sTs = None, None, 0.0
            
            if strMethod == "SDQ":
                with np.errstate(divide='ignore'):
                    sT0 = time.perf_counter()
                    vb, _, _ = obq.iterSequQ(vx, mSigDeltaFilt, 0)
                    sTs = time.perf_counter() - sT0
            
            elif strMethod == "SIGN":
                with np.errstate(divide='ignore'):
                    sT0 = time.perf_counter()
                    vb, _, _ = obq.iterSequQ(vx, mSignFilt, 0)
                    sTs = time.perf_counter() - sT0
            
            elif strMethod == "OBAQ":
                with np.errstate(divide='ignore'):
                    sT0 = time.perf_counter()
                    vb, _, _ = obq.iterSequQ(vx, mfiltMatrix, 0)
                    sTs = time.perf_counter() - sT0    
            
            elif strMethod == "OBBQ_bf16":
                    vb, _, vTb, vLim = obq.iterBlockQ(vx, vw, 16, sPhase = 'lin', sK = None, sType = None, bSilent = True) 
                    
            elif strMethod == "OBBQ_linG16":
                    vb, _, vTb, vLim = obq.iterBlockQ(vx, vw, 16, sPhase = 'lin', sK = None, sType = 'grb', bSilent = True)        
                       
            else:
                raise ValueError(f"Unknown quantization method: '{strMethod}'")

            mb[:, idxBatch] = vb
            if vTb is None:
                vTs[idxBatch] = sTs
            else:
                lstTb.append(vTb)
                lstLim.append(vLim)
            progressBlock.update(idxBatch+1)

        dictQuant[f"mb_{strMethod}"] = mb
        if lstTb:
            dictQuant[f"mTb_{strMethod}"] = np.stack(lstTb, axis=1)
            mLim = np.stack(lstLim, axis=1)
            if mLim.any():
                dictQuant[f"mLim_{strMethod}"] = mLim
        else:
            dictQuant[f"vTs_{strMethod}"] = vTs

    np.savez(sOutPath, **dictQuant)
    print(f"Saved: {sCaseFile}  (methods: {sorted(k[3:] for k in dictQuant if k.startswith('mb_'))})\n")

print("Done.")