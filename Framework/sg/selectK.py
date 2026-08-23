import numpy as np

def selectK(vK, sKRatio=1, strKMode='lin', bReplace=False, rng=None):
    """
    Reduce the number of bins by a ratio.

    vK       -> full bin vector, e.g. from getKFromWD
    sKRatio  -> keep len(vK) // sKRatio bins, 1 -> keep all
    strKMode -> 'lin'  equally spaced over vK
                'rnd'  random draw
    bReplace -> 'rnd' only: True allows the same bin to be drawn more than once
    rng      -> optional np.random.Generator for reproducibility

    Returns
    -------
    vKsel : (K,) ndarray
        Selected bin indices, ascending.
    """
    vK = np.asarray(vK).reshape(-1)

    if sKRatio <= 1:
        return vK

    sNumK = max(int(vK.size // sKRatio), 1)

    if strKMode == 'lin':
        return vK[np.round(np.linspace(0, vK.size - 1, sNumK)).astype(int)]

    if strKMode == 'rnd':
        rng = rng if rng is not None else np.random.default_rng()
        return np.sort(rng.choice(vK, size=sNumK, replace=bReplace))

    raise ValueError(f"unknown strKMode: '{strKMode}'")