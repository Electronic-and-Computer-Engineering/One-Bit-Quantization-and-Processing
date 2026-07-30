#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:46:55 2026

@author: mayerflo
"""
import numpy as np
from scipy import signal as sigP

#import filt

def fir_calcMBKaiser(mWD, sApb, sAsb, sDeltaW,
                     sTaps=None, sMinPhase=False, strWindow='kaiser'):
    """
    Multiband FIR filter design using a Kaiser-window approach.

    Parameters
    ----------
    mWD : (M,2) array_like
        Passband zones in rad/sample, 0 <= w <= pi.
        Each row: [wMin, wMax]. Multiple rows => multiband pass region (union).
    sApb : float
        Passband ripple (dB). Used for Kaiser order estimate.
    sAsb : float
        Stopband attenuation (dB). Used for Kaiser order estimate.
    sDeltaW : float
        Transition width in rad/sample (>= 0). This replaces the old pb/sb edge pairs.
        Example: pi/180, pi/360, ...
    sTaps : int or None
        If None: estimate taps via kaiserord. If given: use exactly this many taps.
    sMinPhase : bool
        If True: convert the FIR to minimum phase using homomorphic method.
    strWindow : str
        'kaiser' (default) or any window name supported by scipy.signal.firwin2.

    Returns
    -------
    vHFilt : ndarray
        FIR filter coefficients.
    vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax : analysis outputs
        Same style as your existing functions (via filt.anFiltKaiser).
        For multiband, the analysis is approximate unless anFiltKaiser supports multiband.
    """

    # --- input checks
    mWD = np.asarray(mWD, dtype=float)
    if mWD.ndim != 2 or mWD.shape[1] != 2:
        raise ValueError("mWD must be of shape (M,2)")

    if np.any(mWD < 0) or np.any(mWD > np.pi):
        raise ValueError("mWD must satisfy 0 <= omega <= pi")

    sDeltaW = float(sDeltaW)
    if sDeltaW <= 0:
        raise ValueError("sDeltaW must be > 0")

    # --- convert omega -> normalized frequency for SciPy in [0,1]
    # f = omega/pi, Nyquist corresponds to 1
    mF = mWD / np.pi

    # --- sort passbands by lower edge
    mF = mF[np.argsort(mF[:, 0])]

    # --- Kaiser order estimate (needs transition width in normalized freq)
    sWidthDig = sDeltaW / np.pi
    if sTaps is None:
        sFiltordK, sBeta = sigP.kaiserord(sAsb - sApb, sWidthDig)
        if sFiltordK % 2 == 0:
            sFiltordK += 1
            print("Found even sFiltordK. Incrementing to odd sFiltord=%d" % sFiltordK)
    else:
        sFiltordK = int(sTaps)
        # if user forces taps, beta is only needed when window is Kaiser
        sBeta = sigP.kaiser_beta(sAsb - sApb) if strWindow.lower() == 'kaiser' else None

    # --- build piecewise-constant desired magnitude response for firwin2
    # we create stop/pass transitions with width sWidthDig around each passband edge
    d = sWidthDig

    vFreq = [0.0]
    vGain = [0.0]

    for fMin, fMax in mF:
        fMin = float(fMin)
        fMax = float(fMax)

        # left transition: stop(0) -> pass(1)
        fL0 = max(0.0, fMin - d)
        fL1 = fMin

        # right transition: pass(1) -> stop(0)
        fR0 = fMax
        fR1 = min(1.0, fMax + d)

        # ensure we stay in stop until fL0
        if fL0 > vFreq[-1]:
            vFreq += [fL0]
            vGain += [0.0]

        # ramp to pass
        vFreq += [fL1]
        vGain += [1.0]

        # hold passband
        vFreq += [fR0]
        vGain += [1.0]

        # ramp to stop
        vFreq += [fR1]
        vGain += [0.0]

    # finish at Nyquist
    if vFreq[-1] < 1.0:
        vFreq += [1.0]
        vGain += [0.0]

    vFreq = np.asarray(vFreq, dtype=float)
    vGain = np.asarray(vGain, dtype=float)

    # remove duplicate frequency points (keep last gain value)
    vF2 = [vFreq[0]]
    vG2 = [vGain[0]]
    for i in range(1, len(vFreq)):
        if vFreq[i] == vF2[-1]:
            vG2[-1] = vGain[i]
        else:
            vF2.append(vFreq[i])
            vG2.append(vGain[i])
    vFreq = np.asarray(vF2, dtype=float)
    vGain = np.asarray(vG2, dtype=float)

    # --- window selection
    if strWindow.lower() == 'kaiser':
        vWindow = ('kaiser', sBeta)
    else:
        vWindow = strWindow

    # --- design FIR
    vHFilt = sigP.firwin2(numtaps=sFiltordK, freq=vFreq, gain=vGain, window=vWindow)

    # --- optional minimum phase
    if sMinPhase:
        vHFilt = sigP.minimum_phase(vHFilt, method='homomorphic')

    # --- analysis (if your anFiltKaiser expects only LP/HP/BP, this is approximate)
    # Here we call it as "multiband" and pass the zones; you can adapt anFiltKaiser accordingly.
    try:
        (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(
            vHFilt, sFs=2.0, strType='multiband', mWD=mWD
        )
    except Exception:
        # If your anFiltKaiser cannot handle multiband yet, return minimal placeholders.
        vw, vH = None, None
        sRpb = sRsb = sHpbMin = sHpbMax = sHsbMax = None

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)






mWD = np.array([[0, 1.1*np.pi/10]])

(vH, vw, vHresp, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = fir_calcMBKaiser(
    mWD=mWD,
    sApb=1,
    sAsb=60,
    sDeltaW=np.pi/45,   # Übergangsbreite
    sTaps=None,
    sMinPhase=False
)

vH_fft           = abs(np.fft.fft(vH))
vHMag        = 20 * np.log10(np.abs(vH_fft) / np.max(abs(vH_fft)))



