import numpy as np

def idealBinFiltFromMW(sN, mW, sValPass=1.0, sValStop=0.0, bFull=True):
    """
    Create an ideal (brick-wall) multiband frequency mask from passband zones mW (omega in [0, pi])
    and return the corresponding time-domain FIR via IFFT.

    Parameters
    ----------
    sN : int
        IFFT length / desired FIR length.
    mW : (M,2) array_like
        Passband zones in omega (rad/sample), 0 <= omega <= pi.
        Each row: [wMin, wMax]. Union of all rows defines passband set.
    sValPass : float
        Mask value inside passbands (default 1.0).
    sValStop : float
        Mask value outside passbands (default 0.0).
    bFull : bool
        If True: enforce conjugate symmetry by mirroring the positive-frequency mask
        to the negative-frequency bins -> real impulse response.
        If False: only fills [0..N/2] and leaves the rest as stop.

    Returns
    -------
    vH : (sN,) ndarray
        FIR coefficients (real).
    vHShift : (sN,) ndarray
        Shifted FIR coefficients (ifftshift), useful for centered linear-phase view.
    vMask : (sN,) ndarray
        Frequency-domain mask (complex-valued array, but real entries).
    """

    sN = int(sN)
    if sN <= 0:
        raise ValueError("sN must be positive")

    mW = np.asarray(mW, dtype=float)
    if mW.ndim == 1:
        if mW.size != 2:
            raise ValueError("If mW is 1D, it must be [wMin, wMax]")
        mW = mW.reshape(1, 2)

    if mW.ndim != 2 or mW.shape[1] != 2:
        raise ValueError("mW must be of shape (M,2)")

    if np.any(mW < 0) or np.any(mW > np.pi):
        raise ValueError("mW must satisfy 0 <= omega <= pi")

    if np.any(mW[:, 0] > mW[:, 1]):
        raise ValueError("each row of mW must satisfy wMin <= wMax")

    # initialize mask
    vMask = np.full(sN, float(sValStop), dtype=float)

    # map omega -> bins on the positive-frequency half [0..N/2]
    # k = round( (omega / (2*pi)) * N )
    for wMin, wMax in mW:
        kMin = int(np.ceil((wMin / (2.0 * np.pi)) * sN))
        kMax = int(np.floor((wMax / (2.0 * np.pi)) * sN))

        # clip to valid positive-frequency indices
        kMin = max(0, min(kMin, sN // 2))
        kMax = max(0, min(kMax, sN // 2))

        if kMax >= kMin:
            vMask[kMin:kMax + 1] = float(sValPass)

    if bFull:
        # enforce real impulse response by conjugate-symmetric magnitude mask:
        # mirror bins 1..N/2-1 to N-1..N/2+1
        # (keep DC=0 and Nyquist=N/2 untouched)
        if sN % 2 == 0:
            # even N: Nyquist bin exists at N/2
            vMask[sN//2 + 1:] = vMask[1:sN//2][::-1]
        else:
            # odd N: no exact Nyquist bin
            vMask[(sN+1)//2:] = vMask[1:(sN+1)//2][::-1]

    # IFFT -> impulse response
    vH = np.fft.ifft(vMask).real

    # centered view (linear-phase FIR typically plotted centered)
    vHShift = np.fft.ifftshift(vH)

    return vH, vHShift, vMask