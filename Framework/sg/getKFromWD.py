import numpy as np

def getKFromWD(mWD, sN):
    """
    Map normalized frequency ranges w_D
    to a vector of integer k indices (on-bin). For creating a signal with on or off bins, this convertion is not relevant. 

    Parameters
    ----------
    mWD : (M,2) array_like
          Each row: [wD_min, wD_max], 0 <= wD <= pi
    sN  : scalar
          DFT length

    Returns
    -------
    vK : (K,) ndarray
         Vector of integer k indices
    """
    mWD = np.asarray(mWD, dtype=float)
    sN = int(sN)
    if sN <= 0:
        raise ValueError("sN must be a positive integer")
        
    if mWD.ndim != 2 or mWD.shape[1] != 2:
        raise ValueError("mWD must be of shape (M, 2)")
        
    if np.any(mWD < 0) or np.any(mWD > np.pi):
        raise ValueError("wD values must satisfy 0 <= wD <= PI")    
        
    if np.any(mWD[:, 0] > mWD[:, 1]):
        raise ValueError("each row of mWD must satisfy wD_min <= wD_max")

    vKList    = []
    sBinPerRad = sN / (2*np.pi)
    for wDmin, wDmax in mWD:
        sKmin = int(np.ceil(sBinPerRad * wDmin))
        sKmax = int(np.floor(sBinPerRad * wDmax))

        if sKmax >= sKmin:
            vKList.append(np.arange(sKmin, sKmax + 1))

    if not vKList:
        return np.array([], dtype=int)

    # concatenate and ensure uniqueness + ordering
    vK = np.unique(np.concatenate(vKList))

    return vK