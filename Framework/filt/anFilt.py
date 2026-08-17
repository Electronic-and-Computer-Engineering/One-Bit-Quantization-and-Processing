import numpy as np
from scipy import signal as sigP

def anFiltKaiser(vHFilt, sFs=None, strType='lowpass',
                 sFpb1=None, sFsb1=None, sFpb2=None, sFsb2=None,
                 mWD=None, sDeltaW=0.0):
    """
    Analyze FIR frequency response on digital frequency grid (rad/sample).

    Notes
    -----
    - Uses vw from scipy.signal.freqz, vw in [0, pi] rad/sample.
    - For multiband: passbands are given by mWD (M,2) in rad/sample.
    - sFs is unused (kept only for backward compatibility).
    """

    vw, vH = sigP.freqz(vHFilt)      # vw in rad/sample, [0, pi]
    vMag = np.abs(vH)

    sDeltaW = float(sDeltaW)
    if sDeltaW < 0:
        raise ValueError("sDeltaW must be >= 0")

    if strType == 'lowpass':
        # sFpb1, sFsb1 are interpreted as omega (rad/sample)
        vPassMask = (vw >= 0) & (vw <= sFpb1)
        vStopMask = (vw >= sFsb1) & (vw <= np.pi)

    elif strType == 'bandpass':
        # sFpb1..sFpb2 pass, [0..sFsb1] and [sFsb2..pi] stop
        vPassMask = (vw >= sFpb1) & (vw <= sFpb2)
        vStopMask = ((vw >= 0) & (vw <= sFsb1)) | ((vw >= sFsb2) & (vw <= np.pi))

    elif strType == 'highpass':
        # pass [sFpb1..pi], stop [0..sFsb1]
        vPassMask = (vw >= sFpb1) & (vw <= np.pi)
        vStopMask = (vw >= 0) & (vw <= sFsb1)

    elif strType == 'multiband':
        if mWD is None:
            raise ValueError("For strType='multiband' you must provide mWD (M,2) in rad/sample.")

        mWD = np.asarray(mWD, dtype=float)
        if mWD.ndim != 2 or mWD.shape[1] != 2:
            raise ValueError("mWD must be of shape (M,2)")
        if np.any(mWD < 0) or np.any(mWD > np.pi):
            raise ValueError("mWD must satisfy 0 <= omega <= pi")

        # passband mask = union of all pass zones
        vPassMask = np.zeros_like(vw, dtype=bool)
        for wMin, wMax in mWD:
            vPassMask |= (vw >= wMin) & (vw <= wMax)

        # transition mask around each edge (optional)
        vTransMask = np.zeros_like(vw, dtype=bool)
        if sDeltaW > 0:
            for wMin, wMax in mWD:
                vTransMask |= (vw >= (wMin - sDeltaW)) & (vw <= (wMin + sDeltaW))
                vTransMask |= (vw >= (wMax - sDeltaW)) & (vw <= (wMax + sDeltaW))

        # stopband = complement of passband, excluding transition zones
        vStopMask = (~vPassMask) & (~vTransMask)

    else:
        raise ValueError("Unsupported filter type. Use 'lowpass', 'bandpass', 'highpass', or 'multiband'.")

    # --- Passband ripple
    vPb = vMag[vPassMask]
    if vPb.size == 0:
        raise ValueError("No frequency samples in passband region(s). Check your edges/mWD.")

    sHpbMin = np.min(vPb)
    sHpbMax = np.max(vPb)
    sRpb = 20 * np.log10(sHpbMax / sHpbMin)

    # --- Stopband attenuation
    vSb = vMag[vStopMask]
    if vSb.size == 0:
        sHsbMax = np.nan
        sRsb = np.nan
    else:
        sHsbMax = np.max(vSb)
        sRsb = -20 * np.log10(sHsbMax) if sHsbMax > 0 else np.inf

    return vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax