#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal as sigP
import filt
import warnings


def fir_calcMBRemez(mWD, sApb, sAsb, sDeltaW,
                    sTaps=None, sMinPhase=False):
    """
    Multiband FIR filter design using the Parks-McClellan (Remez) algorithm.

    Same interface as fir_calcMBKaiser. For a given number of taps Remez
    reaches a considerably higher stopband attenuation, because the error is
    equiripple instead of decaying.

    Parameters
    ----------
    mWD : (M,2) array_like
        Passband zones in rad/sample, 0 <= w <= pi. Each row: [wMin, wMax].
    sApb : float
        Passband ripple (dB) -- sets the pass/stop weighting.
    sAsb : float
        Stopband attenuation (dB) -- sets the pass/stop weighting.
        Remez cannot be told an absolute target; only the RATIO of sApb to
        sAsb matters. What is actually reached follows from sTaps.
    sDeltaW : float
        Transition width in rad/sample (> 0). Transition bands are don't-care.
    sTaps : int or None
        None -> Kaiser order estimate as a starting point.
    sMinPhase : bool
        If True: convert to minimum phase (homomorphic).
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

    # --- omega -> normalized frequency, Nyquist == 1
    mF = mWD / np.pi
    mF = mF[np.argsort(mF[:, 0])]
    if mF.shape[0] > 1 and np.any(mF[1:, 0] <= mF[:-1, 1]):
        raise ValueError("passbands must not overlap")

    sWidthDig = sDeltaW / np.pi

    # --- number of taps
    if sTaps is None:
        sFiltord, _ = sigP.kaiserord(sAsb - sApb, sWidthDig)
    else:
        sFiltord = int(sTaps)

    if sFiltord % 2 == 0:
        sFiltord += 1
        warnings.warn(f"Even number of taps; incrementing to {sFiltord}", stacklevel=2)

    # --- band edges: pass and stop only, transitions are left out
    vBands, vDesired = [], []
    sCursor = 0.0

    for fMin, fMax in mF:
        if fMin > 0.0:
            fL = fMin - sWidthDig
            if fL <= sCursor:
                raise ValueError("transition bands overlap -- reduce sDeltaW "
                                 "or increase the gap between the zones")
            vBands   += [sCursor, fL]
            vDesired += [0.0]

        vBands   += [fMin, fMax]
        vDesired += [1.0]

        sCursor = min(fMax + sWidthDig, 1.0)

    if sCursor < 1.0:
        vBands   += [sCursor, 1.0]
        vDesired += [0.0]

    vBands   = np.asarray(vBands,   dtype=float)
    vDesired = np.asarray(vDesired, dtype=float)

    # --- weighting: only the ratio of the ripples matters
    sDeltaP = (10 ** (sApb / 20) - 1) / (10 ** (sApb / 20) + 1)
    sDeltaS = 10 ** (-sAsb / 20)
    vWeight = np.where(vDesired > 0.5, 1.0, sDeltaP / sDeltaS)

    # --- design FIR
    vHFilt = sigP.remez(sFiltord, vBands, vDesired, weight=vWeight, fs=2.0)

    if sMinPhase:
        vHFilt = sigP.minimum_phase(vHFilt, method='homomorphic')

    # --- analysis
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(
        vHFilt, sFs=2.0, strType='multiband', mWD=mWD)

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)