import numpy as np

def boundRange(vSig, sBound=1.0):
    """
    Scale a real-valued signal symmetrically into [-sBound, +sBound]
    using gain-only normalization (no offset).

    This preserves the signal shape and does not introduce any DC
    component. If the input signal has zero mean, the output will
    also have zero mean (up to scaling).

    Parameters
    ----------
    vSig : array_like
        Input real signal.
    sBound : scalar, optional
        Target symmetric bound (>= 0). Default is 1.0.

    Returns
    -------
    vSigB : ndarray
        Scaled signal with max(abs(vSigB)) <= sBound.
    """
    # ensure numpy array (float)
    vSig = np.asarray(vSig, dtype=float)

    # ensure valid bound
    sBound = float(sBound)
    if sBound < 0:
        raise ValueError("sBound must be >= 0")

    # peak magnitude of input signal
    sMax = np.max(np.abs(vSig))

    # if signal is all zeros, nothing to scale
    if sMax < np.finfo(float).tiny:
        return vSig.copy()

    # gain-only scaling to fit into [-sBound, +sBound]
    return (sBound / sMax) * vSig