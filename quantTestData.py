#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
sys.path.append('01_Library')

import sg, obq, globalTools, filt

# =============================================================================
# SETTINGS
# =============================================================================
sInDir  = "TestBatches"
sOutDir = "QuantBatches"

vCaseFiles = [
    "REAL_FIXED_ONBIN_20260824_124847_679209", #"REAL_FIXED_ONBIN_20260817_181320_775506",     # file names without .npz
]

vMethods = ["OBBQ_sph128"]              # ["SDQ", "OBAQ", "oPWM"]

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
        print(f"  Existing results: {list(dictQuant.keys())}")

    for strMethod in vMethods:

        mb = np.zeros((sN, sBatchSize), dtype=float)       

        for idxBatch in range(sBatchSize):
            if idxBatch == 0:
                progressBlock = globalTools.SimpleProgressBar(sBatchSize, width=40, prefix = strMethod, fill="█", empty=" ", end=" ✓")                       

            vx = mx[:, idxBatch]

            if strMethod == "SDQ":
                with np.errstate(divide='ignore'):
                    vb, _, _ = obq.iterSequQ(vx, mSigDeltaFilt, 0)
                    
            elif strMethod == "SIGN":
                with np.errstate(divide='ignore'):
                    vb, _, _ = obq.iterSequQ(vx, mSignFilt, 0) 

            elif strMethod == "OBAQ":
                with np.errstate(divide='ignore'):
                    vb, _, _ = obq.iterSequQ(vx, mfiltMatrix, 0)      
            
            elif strMethod == "OBBQ_bf16":
                    vb, _ = obq.iterBlockQ(vx, vw, 16, sPhase = 'lin', sK = None, sType = None, bSilent = True) 
                    
            elif strMethod == "OBBQ_linG16":
                    vb, _ = obq.iterBlockQ(vx, vw, 16, sPhase = 'lin', sK = None, sType = 'grb', bSilent = True)        
           
            elif strMethod == "OBBQ_bf32":
                    vb, _ = obq.iterBlockQ(vx, vw, 32, sPhase = 'lin', sK = None, sType = None, bSilent = True) 
                    
            elif strMethod == "OBBQ_linG32":
                    vb, _ = obq.iterBlockQ(vx, vw, 32, sPhase = 'lin', sK = None, sType = 'grb', bSilent = True)   
                    
            elif strMethod == "OBBQ_bf64":
                    vb, _ = obq.iterBlockQ(vx, vw, 64, sPhase = 'lin', sK = None, sType = None, bSilent = True) 
                    
            elif strMethod == "OBBQ_linG64":
                    vb, _ = obq.iterBlockQ(vx, vw, 64, sPhase = 'lin', sK = None, sType = 'grb', bSilent = True)           
            
            elif strMethod == "OBBQ_sph16":
                    vb, _ = obq.iterBlockQ(vx, vw, 16, sPhase = 'lin', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sph32":
                    vb, _ = obq.iterBlockQ(vx, vw, 32, sPhase = 'lin', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sph64":
                    vb, _ = obq.iterBlockQ(vx, vw, 64, sPhase = 'lin', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sph128":
                    vb, _ = obq.iterBlockQ(vx, vw, 128, sPhase = 'lin', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sphMin16":
                    vb, _ = obq.iterBlockQ(vx, vwMin, 16, sPhase = 'min', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sphMin32":
                    vb, _ = obq.iterBlockQ(vx, vwMin, 32, sPhase = 'min', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sphMin64":
                    vb, _ = obq.iterBlockQ(vx, vwMin, 64, sPhase = 'min', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_sphMin128":
                    vb, _ = obq.iterBlockQ(vx, vwMin, 128, sPhase = 'min', sK = None, sType = 'sphere', bSilent = True)

            elif strMethod == "OBBQ_tabu":
                    vb, _ = obq.iterBlockQ(vx, vwMin, 16, sPhase = 'min', sK = None, sType = 'tabu', bSilent = True)
                     
            else:
                raise ValueError(f"Unknown quantization method: '{strMethod}'")

            mb[:, idxBatch] = vb
            progressBlock.update(idxBatch+1)

        dictQuant[f"mb_{strMethod}"] = mb
        print(f"  [{strMethod}] done -> key: mb_{strMethod}")

    np.savez(sOutPath, **dictQuant)
    print(f"Saved: {sCaseFile}  (keys: {list(dictQuant.keys())})\n")

print("Done.")