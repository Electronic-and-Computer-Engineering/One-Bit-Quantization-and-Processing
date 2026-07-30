import os
import numpy as np

def writeCaseMarkdown(caseDir, strCaseName, dCase):
    """
    Create README.md inside caseDir containing all case settings.
    """

    lines = []
    lines.append(f"# Case: {dCase['caseCode']}\n")

    # --- scalar settings
    lines.append("## Core settings")
    lines.append(f"- sN: {dCase['sN']}")
    lines.append(f"- sL: {dCase['sL']}")
    lines.append(f"- sBatchSize: {dCase['sBatchSize']}")
    lines.append(f"- sBeta: {dCase['sBeta']}")
    lines.append(f"- sBound: {dCase['sBound']}\n")

    # --- band matrices
    lines.append("## Band definitions")

    mWD = np.asarray(dCase["mWD"])
    mR  = np.asarray(dCase["mR"])

    lines.append("### mWD (signal bands)")
    for row in mWD:
        lines.append(f"- [{row[0]:.6f}, {row[1]:.6f}]")

    lines.append("\n### mR (reconstruction bands)")
    for row in mR:
        lines.append(f"- [{row[0]:.6f}, {row[1]:.6f}]")

    # --- kaiser
    k = dCase["kaiser"]
    lines.append("\n## Kaiser filter settings")
    lines.append(f"- sApb: {k['sApb']}")
    lines.append(f"- sAsb: {k['sAsb']}")
    lines.append(f"- sDeltaW: {k['sDeltaW']}")
    lines.append(f"- bMinPhase: {k['bMinPhase']}")

    # --- write file
    readmePath = os.path.join(caseDir, f"{strCaseName}.md")
    with open(readmePath, "w") as f:
        f.write("\n".join(lines))