#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:43:05 2026

@author: mayerflo
"""
import os

def writeEvalMarkdown(sPath, sCaseFile, dictRes, vKeys, sPathCaseMd):
    """Batch means as a table, followed by the case settings of stage 1."""
    vLines = [f"# {sCaseFile}", "",
              "## Results (batch means)", "",
              "| method | recon | MSE | SNR (dB) | PSNR (dB) |",
              "|---|---|---|---|---|"]

    for k in vKeys:
        strRecon, strMethod = k.split("_", 1)
        vMean = dictRes[k].mean(axis=0)
        vLines.append(f"| {strMethod} | {strRecon} | "
                      f"{vMean[0]:.4g} | {vMean[1]:.2f} | {vMean[2]:.2f} |")

    vLines += ["", "## Case settings", ""]
    if os.path.exists(sPathCaseMd):
        with open(sPathCaseMd) as f:
            vLines += f.read().splitlines()[1:]      # drop the original heading

    with open(sPath, "w") as f:
        f.write("\n".join(vLines) + "\n")