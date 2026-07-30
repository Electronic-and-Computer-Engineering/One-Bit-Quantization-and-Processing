from datetime import datetime
import re

def makeCaseName(dCase):
    caseCode = str(dCase["caseCode"]).upper()

    # sanitize
    caseCode = re.sub(r"[^A-Z0-9_]+", "_", caseCode)
    caseCode = re.sub(r"_+", "_", caseCode).strip("_")

    sN = int(dCase["sN"])
    sL = int(dCase["sL"])

    # short beta tag only if off-bin
    sBeta = float(dCase.get("sBeta", 0.0))
    betaTag = "" if abs(sBeta) < 1e-12 else f"_B{str(sBeta).replace('.','p')}"

    # timestamp → uniqueness
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{caseCode}_N{sN}_L{sL}{betaTag}_{ts}"