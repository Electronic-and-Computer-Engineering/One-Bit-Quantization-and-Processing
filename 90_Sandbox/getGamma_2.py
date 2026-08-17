# %%
# system packages
import numpy as np
import matplotlib.pyplot as mtplt
import scipy.linalg as scLinAlg
import scipy.signal as sigP
from scipy.fftpack import hilbert
import sys, os, json
from datetime import datetime   # <— hinzugefügt

# ----------------------------
# Logging aktivieren (Tee-Logger)
# ----------------------------
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        # line-buffered schreiben, damit die Datei laufend aktualisiert wird
        self.log = open(filename, "w", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sTimestamp  = datetime.now().strftime("%y%m%d_%H%M")
sLogFname   = f"WindowTest_{sTimestamp}.txt"
sys.stdout  = TeeLogger(sLogFname)
sys.stderr  = sys.stdout   # Fehlerausgaben auch mitschreiben
print(f"[LOGGER] Writing console output to: {sLogFname}")

mtplt.rcParams['mathtext.fontset'] = 'stix'
mtplt.rcParams['font.family'] = 'STIXGeneral'
mtplt.close('all')
np.seterr(all='ignore')

# %%
# Library-Pfade

sys.path.append('../01_Library')

import sg, sa, sp, obq, filt
import globalTools

# ----------------------------
# Setup (deine Parameter)
# ----------------------------
sNbins    = 2048
sM        = 32
sL_max    = sNbins
sHop      = 8
sK        = 213
sBSize    = 8

# ----------------------------
# Signal-Setup (dein Code)
# ----------------------------
sNewSignal = False  # wenn True -> neu generieren und speichern
if sNewSignal:
    sFs = 2048
    sSigFmax = 213
    vxFrequ = (np.arange(1, sSigFmax, step=1)).reshape(-1, 1)
    vxPhase = np.random.rand(len(vxFrequ), 1) * 2 * np.pi
    os.makedirs('saves', exist_ok=True)
    np.save('saves/vxFrequ.npy', vxFrequ)
    np.save('saves/vxPhase.npy', vxPhase)
    np.save('saves/sSigFmax.npy', sSigFmax)
    np.save('saves/sFs.npy', sFs)
else:
    vxFrequ  = np.load('saves/vxFrequ.npy')
    vxPhase  = np.load('saves/vxPhase.npy')
    sSigFmax = int(np.load('saves/sSigFmax.npy'))
    sFs      = int(np.load('saves/sFs.npy'))

sT = 1 / (sFs)
v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# ----------------------------
# Filter-Setup (dein Code)
# ----------------------------
vLPRangeFs        = [0, 0, sSigFmax, sSigFmax]
vLPfilterRad      = sg.freq2rad(vLPRangeFs, sFs)
vLPfilterRadIdeal = sg.freq2rad(vLPRangeFs, sFs)

vBPRangeFs   = [180, 200, 311, 331]
vBPRangeRad  = sg.freq2rad(vBPRangeFs, sFs)
mOptFilterRange = np.vstack((vLPfilterRad, vBPRangeRad))
vMaxVal      = [1, 0.5]

# ideale Matrizen
vRIdeal, vRIdealShift, vW = filt.idealBinFilt(sNbins, vLPfilterRadIdeal, 1.0, 0.0, True)
mRIdeal = scLinAlg.toeplitz(vRIdeal)

# Entwurfsfilter (nur für Auswertung)
sFpb, sFsb, sApbdB, sAsbdB = sSigFmax, sSigFmax + 40, 0.001, 60
(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(
    sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False
)
os.makedirs('saves', exist_ok=True)
np.save('saves/vWcoeff.npy', vWcoeff)

# DFT-Gewichtungen
vFiltIdeal, _ , vW = filt.idealBinFilt(sNbins, vLPfilterRad, vMaxVal[0], 0, True)
vW_D = sp.getFiltWeights(vW, sK, vLPfilterRad)

# ----------------------------
# Fallback: Dual-Gamma Fenster (falls im sg nicht vorhanden)
# ----------------------------
def modHammingDualGamma(M: int,
                        c: float,
                        rL: float = 0.08,
                        rR: float = 0.08,
                        gammaL: float = 1.0,
                        gammaR: float = 1.0,
                        l2_norm: bool = False):
    assert M >= 2
    assert 0.0 < rL < 1.0 and 0.0 < rR < 1.0
    c = float(np.clip(c, 0.0, M - 1.0))
    n = np.arange(M, dtype=float)
    midL = max(c, 1e-12)
    midR = max((M - 1.0) - c, 1e-12)
    tL = np.clip((c - n) / midL, 0.0, 1.0)
    tR = np.clip((n - c) / midR, 0.0, 1.0)
    hL = 0.5 * (1.0 + np.cos(np.pi * tL))
    hR = 0.5 * (1.0 + np.cos(np.pi * tR))
    hL = hL ** float(gammaL)
    hR = hR ** float(gammaR)
    wL = rL + (1.0 - rL) * hL
    wR = rR + (1.0 - rR) * hR
    w = np.where(n <= c, wL, wR)
    if l2_norm:
        w *= (np.sqrt(M) / max(np.linalg.norm(w), 1e-12))
    return w

# ----------------------------
# Bewertung (dein Ideal-Filter-Setup)
# ----------------------------
def eval_SNR_Ideal(vx_in: np.ndarray, vb_out: np.ndarray) -> float:
    vxSigFiltIdeal     = mRIdeal @ vx_in
    vbSequErrFiltIdeal = mRIdeal @ (vx_in - vb_out)
    _, sVB_SNRdbIdeal, _ = sa.evalN(vbSequErrFiltIdeal, vxSigFiltIdeal)
    return float(sVB_SNRdbIdeal)

# ----------------------------
# Runner mit sauberem Patch von sg.modHamming
# ----------------------------
def run_with_window_params(vx_in: np.ndarray,
                           rL: float, rR: float, gammaL: float, gammaR: float,
                           time_limit_block: float = 0.5,
                           verbose: bool = False):
    """
    Führt DEINE iterBlockQDFT aus, ohne deren Code zu ändern.
    Patcht nur sg.modHamming, zwingt rL/rR/gammaL/gammaR.
    Stellt am Ende die Originalfunktion immer wieder her.
    """
    # Original sichern
    modHam_orig = sg.modHamming

    # Patch definieren
    def _patched_modHamming(M, c, **kwargs):
        l2n = kwargs.get('l2_norm', False)
        try:
            return modHam_orig(M=M, c=c, rL=rL, rR=rR,
                               gamma=(gammaL, gammaR), l2_norm=l2n)
        except TypeError:
            return modHammingDualGamma(M=M, c=c, rL=rL, rR=rR,
                                       gammaL=gammaL, gammaR=gammaR, l2_norm=l2n)

    # Patch anwenden
    sg.modHamming = _patched_modHamming

    try:
        vb_out = obq.iterBlockQDFT(
            vx_in, vLPfilterRad, vFiltIdeal, sK, sM, sL_max, sHop,
            'grb', verbose, 1
        )
    finally:
        sg.modHamming = modHam_orig

    snr_db = eval_SNR_Ideal(vx_in, vb_out)
    if verbose:
        print(f"SNR_ideal = {snr_db:.3f} dB  (rL={rL}, rR={rR}, gL={gammaL}, gR={gammaR})")
    return vb_out, snr_db

# ----------------------------
# Grid-Search über Fensterparameter
# ----------------------------
v_rL     = [0.95, 0.98]
v_rR     = [0.10, 0.15, 0.20]
v_gammaL = [0.15, 0.20, 0.25]
v_gammaR = [0.90, 1.00, 1.1]

best = {"rL": None, "rR": None, "gammaL": None, "gammaR": None, "SNR_dB": -np.inf}
best_vb = None

print("=== Grid-Search Fensterparameter ===")
for sRL in v_rL:
    for sRR in v_rR:
        for sGL in v_gammaL:
            for sGR in v_gammaR:
                vb_tmp, sSNR = run_with_window_params(
                    vx, rL=sRL, rR=sRR, gammaL=sGL, gammaR=sGR,
                    time_limit_block=0.5, verbose=False
                )
                if sSNR > best["SNR_dB"]:
                    best.update({"rL": sRL, "rR": sRR, "gammaL": sGL, "gammaR": sGR, "SNR_dB": sSNR})
                    best_vb = vb_tmp
                print(f"rL={sRL:.2f}, rR={sRR:.2f}, gL={sGL:.2f}, gR={sGR:.2f}  ->  {sSNR:6.3f} dB"
                      f"{'   (NEW BEST)' if sSNR==best['SNR_dB'] else ''}")

print("\n=== Beste Fenster-Parameter ===")
print(best)

# ----------------------------
# Ergebnisse speichern
# ----------------------------
os.makedirs('saves', exist_ok=True)
np.save('saves/best_window_params.npy',
        np.array([best["rL"], best["rR"], best["gammaL"], best["gammaR"], best["SNR_dB"]], dtype=float))
np.save('saves/vBDFT_best.npy', best_vb)
with open('saves/best_window_params.json', 'w') as f:
    json.dump(best, f, indent=2)

print("\nGespeichert:")
print("  saves/best_window_params.npy")
print("  saves/best_window_params.json")
print("  saves/vBDFT_best.npy")

# ----------------------------
# Optional: finale Auswertung (dein Stil)
# ----------------------------
vX = np.fft.fft(mRIdeal @ vx, sNbins)
vBfft = np.fft.fft(best_vb, sNbins)
vDiffSingle = vX - vBfft
vSingleRec = np.fft.fft(mRIdeal @ best_vb, sNbins)

vXMag          = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))
vBfftMag       = 20*sa.safelog10(np.abs(vBfft) / np.max(abs(vX)))
vDiffSingleMag = 20*sa.safelog10(np.abs(vDiffSingle) / np.max(abs(vX)))
vSingleRecMag  = 20*sa.safelog10(np.abs(vSingleRec) / np.max(abs(vX)))

print(f"\nFinal SNR_ideal (best): {best['SNR_dB']:.3f} dB")