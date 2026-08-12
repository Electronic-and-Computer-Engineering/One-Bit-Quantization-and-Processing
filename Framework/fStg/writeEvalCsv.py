#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:42:00 2026

@author: mayerflo
"""

import numpy as np

def writeEvalCsv(sPath, dictRes, vKeys, sBatchSize):
    """One row per signal, three columns (MSE, SNR, PSNR) per key."""
    vHeader = ["idx"] + [f"{k}_{m}" for k in vKeys for m in ("MSE","SNR","PSNR")]
    mTable  = np.column_stack([np.arange(sBatchSize)] + [dictRes[k] for k in vKeys])
    np.savetxt(sPath, mTable, delimiter=",", header=",".join(vHeader),
               comments="", fmt=["%d"] + ["%.6g"] * (mTable.shape[1] - 1))