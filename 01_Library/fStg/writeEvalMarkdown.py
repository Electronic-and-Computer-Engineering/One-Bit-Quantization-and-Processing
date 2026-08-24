#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:43:05 2026
@author: mayerflo
"""
import os
from datetime import datetime

# --- Dublin Core fields that are identical for every evaluation
DICT_DC = {
    "creator":   "Mayer, Florian",
    "publisher": "FH JOANNEUM",
    "rights":    "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
    "language":  "en",
    "subject":   "one-bit quantization; noise shaping; signal processing; "
                 "quantization error; signal-to-error ratio",
    "source":    "analyzeTestData.py",
}

def readCaseBody(sPathCaseMd):
    """Case settings of stage 1, without its Dublin Core block and heading."""
    if not os.path.exists(sPathCaseMd):
        return []

    with open(sPathCaseMd) as f:
        vSrc = f.read().splitlines()

    # skip the YAML front matter, if present
    if vSrc and vSrc[0].strip() == "---":
        idxEnd = next((i for i in range(1, len(vSrc))
                       if vSrc[i].strip() == "---"), 0)
        vSrc = vSrc[idxEnd + 1:]

    # skip leading blanks and the '# <case>' heading
    while vSrc and (not vSrc[0].strip() or vSrc[0].startswith("# ")):
        vSrc = vSrc[1:]

    return vSrc


def writeEvalMarkdown(sPath, sCaseFile, dictRes, vKeys, sPathCaseMd):
    """
    Batch means as a table, followed by the case settings of stage 1.

    The file starts with a Dublin Core metadata block (ISO 15836, 15 elements)
    as required by the data management plan.
    """
    vMethods   = sorted({k.split("_", 1)[1] for k in vKeys})
    sBatchSize = dictRes[vKeys[0]].shape[0]

    vLines = ["---",
              f"dc.title:       {sCaseFile} -- evaluation",
              f"dc.creator:     {DICT_DC['creator']}",
              f"dc.subject:     {DICT_DC['subject']}",
              f"dc.description: MSE, SNR and PSNR of {len(vMethods)} "
              f"quantization methods ({', '.join(vMethods)}) over "
              f"{sBatchSize} realizations, for ideal and non-ideal "
              f"reconstruction. Batch means listed below.",
              f"dc.publisher:   {DICT_DC['publisher']}",
              "dc.contributor: ",
              f"dc.date:        {datetime.now().isoformat(timespec='seconds')}",
              "dc.type:        Dataset",
              "dc.format:      text/csv; application/octet-stream (NumPy .npz); "
              "text/markdown",
              f"dc.identifier:  {sCaseFile}_eval",
              f"dc.source:      {DICT_DC['source']}",
              f"dc.language:    {DICT_DC['language']}",
              f"dc.relation:    {sCaseFile}.npz (signals); "
              f"{sCaseFile}_eval.npz; {sCaseFile}_eval.csv",
              f"dc.coverage:    {sBatchSize} realizations, "
              f"{len(vKeys)} metric sets",
              f"dc.rights:      {DICT_DC['rights']}",
              "---", "",
              f"# {sCaseFile}", "",
              "## Results (batch means)", "",
              "| method | recon | MSE | SNR (dB) | PSNR (dB) |",
              "|---|---|---|---|---|"]

    for k in vKeys:
        strRecon, strMethod = k.split("_", 1)
        vMean = dictRes[k].mean(axis=0)
        vLines.append(f"| {strMethod} | {strRecon} | "
                      f"{vMean[0]:.4g} | {vMean[1]:.2f} | {vMean[2]:.2f} |")

    vLines += [""] + readCaseBody(sPathCaseMd)

    with open(sPath, "w") as f:
        f.write("\n".join(vLines) + "\n")

    return sPath