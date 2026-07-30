import numpy as np

def modHamming(M: int,
               c: float,
               rL: float = 0.08,
               rR: float = 0.08,
               gammaL: float = 1.0,
               gammaR: float = 1.0,
               l2_norm: bool = False):
    """
    Asymmetric 'Hamming-like' window of length M with peak at position c.
    Left/right edges hit rL / rR exactly. Shape on each side is a half raised-cosine
    with curvature controlled independently by gammaL and gammaR.

    Args:
        M      : window length (>=2)
        c      : desired peak position in [0, M-1]
        rL     : left edge value w[0]  in (0,1)
        rR     : right edge value w[M-1] in (0,1)
        gammaL : curvature left side  (<1 flatter, >1 steeper)
        gammaR : curvature right side (<1 flatter, >1 steeper)
        l2_norm: if True, normalize to ||w||_2 = sqrt(M)

    Returns:
        w      : np.ndarray, shape (M,)
    """
    assert M >= 2, "M must be >= 2"
    assert 0.0 < rL < 1.0 and 0.0 < rR < 1.0, "rL/rR must be in (0,1)"

    c = float(np.clip(c, 0.0, M - 1.0))
    n = np.arange(M, dtype=float)

    midL = max(c, 1e-12)                # distance from left edge to center
    midR = max((M - 1.0) - c, 1e-12)    # distance from center to right edge

    # normalized distances to center
    tL = np.clip((c - n) / midL, 0.0, 1.0)  # 0 at center, 1 at left edge
    tR = np.clip((n - c) / midR, 0.0, 1.0)  # 0 at center, 1 at right edge

    # half raised-cosine shape
    hL = 0.5 * (1.0 + np.cos(np.pi * tL))
    hR = 0.5 * (1.0 + np.cos(np.pi * tR))

    # apply curvature separately
    hL = hL ** gammaL
    hR = hR ** gammaR

    # map to exact edge levels
    wL = rL + (1.0 - rL) * hL
    wR = rR + (1.0 - rR) * hR

    # combine left/right sides
    w = np.where(n <= c, wL, wR)

    if l2_norm:
        w *= (np.sqrt(M) / max(np.linalg.norm(w), 1e-12))

    return w