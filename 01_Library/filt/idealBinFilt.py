import numpy as np

def idealBinFilt(sN, mW, sValPass=1.0, sValStop=0.0, bFull=True):
    """
    mW : (M,4) array_like
        Per band: [wStop1, wPass1, wPass2, wStop2] in rad/sample.
        Linear ramp from sValStop to sValPass between wStop1 and wPass1.
        Flat sValPass between wPass1 and wPass2.
        Linear ramp from sValPass to sValStop between wPass2 and wStop2.
    """
    sN = int(sN)
    mW = np.asarray(mW, dtype=float)
    if mW.ndim == 1:
        mW = mW.reshape(1, 4)
    
    vMask = np.full(sN, float(sValStop), dtype=float)
    
    for wStop1, wPass1, wPass2, wStop2 in mW:
        kStop1 = int(np.ceil((wStop1 / (2*np.pi)) * sN))
        kPass1 = int(np.floor((wPass1 / (2*np.pi)) * sN))
        kPass2 = int(np.floor((wPass2 / (2*np.pi)) * sN))
        kStop2 = int(np.floor((wStop2 / (2*np.pi)) * sN))

        kStop1 = max(0, min(kStop1, sN//2))
        kPass1 = max(0, min(kPass1, sN//2))
        kPass2 = max(0, min(kPass2, sN//2))
        kStop2 = max(0, min(kStop2, sN//2))

        # Rising ramp: wStop1 -> wPass1
        if kPass1 > kStop1:
            vMask[kStop1:kPass1] = np.linspace(sValStop, sValPass, kPass1-kStop1)

        # Flat passband: wPass1 -> wPass2
        if kPass2 >= kPass1:
            vMask[kPass1:kPass2+1] = sValPass

        # Falling ramp: wPass2 -> wStop2
        if kStop2 > kPass2:
            vMask[kPass2:kStop2] = np.linspace(sValPass, sValStop, kStop2-kPass2)

    if bFull:
        if sN % 2 == 0:
            vMask[sN//2+1:] = vMask[1:sN//2][::-1]
        else:
            vMask[(sN+1)//2:] = vMask[1:(sN+1)//2][::-1]

    vH      = np.fft.ifft(vMask).real
    vHShift = np.fft.ifftshift(vH)
    return vH, vHShift, vMask