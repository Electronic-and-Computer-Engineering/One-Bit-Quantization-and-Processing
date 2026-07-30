import numpy as np

def saveScen2txt(vx, J, s, sM, sN, sHop, sFs, sFmax, dft_snr, iscas_snr, filename="szenarien.txt"):
    """
    Hängt ein neues Szenario mit vx-Vektor und SNR-Werten an eine Textdatei an.

    Parameters
    ----------
    vx : array-like
        Dein Vektor (z. B. numpy array oder Liste).
    dft_snr : float
        Ergebnis für DFTQ (SNR in dB).
    iscas_snr : float
        Ergebnis für ISCAS25 (SNR in dB).
    filename : str
        Pfad zur Textdatei (default: 'szenarien.txt').
    """
    with open(filename, "a") as f:
        f.write("Neues Szenario:\n")
        f.write(f"sM: {sM}, sN: {sN}, sHop: {sHop}, sFs: {sFs}, sFmax: {sFmax}" + "\n")
        f.write("vx: " + ", ".join([f"{v:.6e}" for v in vx]) + "\n")
        f.write("J: " + ", ".join([f"{j:.6e}" for j in J]) + "\n")
        f.write(f"According to J index (within Hop) with lowest edge energy: {s}" + "\n")
        f.write(f"DFTQ: {dft_snr:.2f} dB\n")
        f.write(f"ISCAS25: {iscas_snr:.2f} dB\n")
        f.write("=============\n\n")
