import numpy as np
import scipy.signal as sigp


# === Analyse- und Synthese-Funktionen ===
def OLA_analysis(vx, sM, sH, window_type='hann', sPad=None, ApplyWindow=True):
    vx = np.asarray(vx).flatten()
    if sPad is None:
        sPad = sM // 2

    vx_pad = np.pad(vx, (sPad, sPad), mode = 'constant', constant_values=(0, 0))
    N_pad = len(vx_pad)
    vWin = sigp.get_window(window_type, sM)

    numBlocks = (N_pad - sM) // sH + 1
    mBlocks = np.zeros((sM, numBlocks))

    for p in range(numBlocks):
        sStart = p * sH
        vSeg = vx_pad[sStart:sStart + sM]
        if ApplyWindow:
            vSeg = vSeg * vWin
        mBlocks[:, p] = vSeg

    return mBlocks, vWin, vx_pad


def OLA_synth(mBlocks, sM, sH, window_type='hann', sPad=None, ApplyDePadding=True):
    sM = int(sM)
    sH = int(sH)
    numBlocks = mBlocks.shape[1]
    if sPad is None:
        sPad = sM // 2

    vWin = sigp.get_window(window_type, sM)
    sNSynth = (numBlocks - 1) * sH + sM
    vSynth = np.zeros(sNSynth)
    vNnorm = np.zeros(sNSynth)

    for p in range(numBlocks):
        sStart = p * sH
        vSynth[sStart:sStart + sM] += mBlocks[:, p] * vWin
        vNnorm[sStart:sStart + sM] += vWin ** 2

    #vSynth /= (vNnorm + 1e-16)

    if ApplyDePadding:
        vSynth = vSynth[sPad:-sPad]

    return vSynth

def olaAddwin(sM, sWLen, sHop, fWinFn = np.hanning, sNorm = True):
    
    """
    Computes the OLA weight function for a block of length M.
    One window is placed centered on M, then ov copies are placed
    to the left and right, each shifted by Hop samples.

    Works correctly for both even and odd winLen.

    Args:
        sM          : block length in samples
        sWLen       : window length in samples (even or odd)
        sHop        : hop size in samples
        sWinFn      : window function, e.g. np.hanning, np.hamming, np.blackman
        sNorm       : if True, normalize OLA sum to peak = 1

    Returns:
        ola_sum     : full OLA weight array (length = total_len)
        sBlockStart : index of first sample of M in ola_sum
        sOv          : number of shifted copies per side
    """
    sOv        = int(np.floor(((sWLen - sM) / 2 + sM) / sHop)) - 1
    sHalfWin   = sWLen // 2

    sBlockCent   = sOv * sHop + sHalfWin
    sBlockStart  = sBlockCent - sM // 2
    total_len    = sBlockCent + sOv * sHop + sHalfWin + (1 if sWLen % 2 == 0 else 1)

    w_base  = fWinFn(sWLen)
    vOlaSum = np.zeros(total_len)

    for i in range(-sOv, sOv + 1):
        sCenter     = sBlockCent + i * sHop
        idxStart    = sCenter - sHalfWin
        idxEnd      = idxStart + sWLen
        vOlaSum[idxStart:idxEnd] += w_base

    if sNorm:
        sPeak = np.max(vOlaSum)
        if sPeak > 0:
            vOlaSum /= sPeak

    return vOlaSum, sBlockStart, sOv