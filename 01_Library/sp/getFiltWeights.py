import numpy as np
from scipy.interpolate import interp1d

def getFiltWeights(vW, vK, mRange):
    vW     = np.asarray(vW)
    mRange = np.asarray(mRange)

    # One-sided frequency axis [0, pi)
    vOmegaVw = np.linspace(0, np.pi, len(vW) // 2, endpoint=False)
    vW_half  = vW[:len(vW) // 2]

    if mRange.ndim == 1:
        mRange = mRange[np.newaxis, :]

    f_interp = interp1d(vOmegaVw, vW_half,
                        kind='linear',
                        bounds_error=False,
                        fill_value=0.0)
    vW_D = []
    for band in mRange:
        omega_lo      = band[0]              # PassStart
        omega_hi      = band[3]              # PassEnd
        vOmega_target = np.linspace(omega_lo, omega_hi, vK)
        vW_D.append(f_interp(vOmega_target))

    return np.hstack(vW_D)                  # shape: (B*vK,) oder (vK,)