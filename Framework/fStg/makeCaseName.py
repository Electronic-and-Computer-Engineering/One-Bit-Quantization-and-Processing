from datetime import datetime

def makeCaseCode(dCase):
    strSig = "REAL"  if dCase["strSig"] == "real" else "CPLX"
    strAmp = "FIXED" if dCase.get("vAmp") is None else "RNDAMP"
    strBin = "ONBIN" if dCase["sBeta"] == 0       else "OFFBIN"
    return f"{strSig}_{strAmp}_{strBin}"

def makeCaseName(dCase):
    return f"{makeCaseCode(dCase)}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"    