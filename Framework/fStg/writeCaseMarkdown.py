import os
import numpy as np


def fmtVal(val):
    """Format a single value as a one-line string."""
    if isinstance(val, np.ndarray):
        return "[" + "; ".join(", ".join(f"{x:.6f}" for x in np.atleast_1d(row))
                               for row in np.atleast_2d(val)) + "]"
    return str(val)


def writeCaseMarkdown(strDir, strCaseName, dCase):
    """
    Write strDir/<strCaseName>.md containing all fields of dCase.

    The dict is iterated instead of using a fixed field list, so new
    entries in vCases show up in the markdown automatically and stay
    discoverable via full-text search.
    """
    vLines = [f"# {strCaseName}", ""]

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