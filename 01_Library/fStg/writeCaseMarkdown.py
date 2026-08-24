import os
import numpy as np
from datetime import datetime

# --- Dublin Core fields that are identical for every case
DICT_DC = {
    "creator":   "Mayer, Florian",
    "publisher": "FH JOANNEUM",
    "rights":    "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
    "language":  "en",
    "subject":   "one-bit quantization; noise shaping; signal processing; test signals",
    "source":    "genTestSignals.py",
}


def fmtVal(val):
    """Format a single value as a one-line string."""
    if isinstance(val, np.ndarray):
        return "[" + "; ".join(", ".join(f"{x:.6f}" for x in np.atleast_1d(row))
                               for row in np.atleast_2d(val)) + "]"
    return str(val)


def fmtCoverage(dCase):
    """Dublin Core 'coverage': spectral and temporal extent of the batch."""
    vParts = []
    for strKey, strName in (("mWX", "signal"),
                            ("mWD", "shaping"),
                            ("mR",  "reconstruction")):
        if strKey in dCase:
            mBand = np.atleast_2d(dCase[strKey])
            vParts.append(f"{strName} " +
                          "; ".join(f"{r[0]:.6f}-{r[1]:.6f}" for r in mBand) +
                          " rad/sample")
    vParts.append(f"{dCase['sN']} samples")
    return ", ".join(vParts)


def writeCaseMarkdown(strDir, strCaseName, dCase):
    """
    Write strDir/<strCaseName>.md containing all fields of dCase.

    The file starts with a Dublin Core metadata block (ISO 15836, 15 elements)
    as required by the data management plan, followed by the case settings.
    The dict is iterated instead of using a fixed field list, so new entries
    in vCases show up in the markdown automatically and stay discoverable via
    full-text search.
    """
    vLines = ["---",
              f"dc.title:       {strCaseName}",
              f"dc.creator:     {DICT_DC['creator']}",
              f"dc.subject:     {DICT_DC['subject']}",
              f"dc.description: Synthetic multi-tone test signal batch, "
              f"{dCase.get('sBatchSize', '?')} realizations of length "
              f"{dCase.get('sN', '?')}. Parameters listed below.",
              f"dc.publisher:   {DICT_DC['publisher']}",
              "dc.contributor: ",
              f"dc.date:        {datetime.now().isoformat(timespec='seconds')}",
              "dc.type:        Dataset",
              "dc.format:      application/octet-stream (NumPy .npz); text/markdown",
              f"dc.identifier:  {strCaseName}",
              f"dc.source:      {DICT_DC['source']}",
              f"dc.language:    {DICT_DC['language']}",
              f"dc.relation:    {strCaseName}.npz",
              f"dc.coverage:    {fmtCoverage(dCase)}",
              f"dc.rights:      {DICT_DC['rights']}",
              "---", "",
              f"# {strCaseName}", "",
              "## Case settings", ""]

    for strKey, val in dCase.items():
        if isinstance(val, dict):
            vLines.append(f"- {strKey}:")
            for strSub, subVal in val.items():
                vLines.append(f"    - {strSub}: {fmtVal(subVal)}")
        else:
            vLines.append(f"- {strKey}: {fmtVal(val)}")

    vLines.append("")

    strPath = os.path.join(strDir, strCaseName + ".md")
    with open(strPath, "w") as f:
        f.write("\n".join(vLines))

    return strPath