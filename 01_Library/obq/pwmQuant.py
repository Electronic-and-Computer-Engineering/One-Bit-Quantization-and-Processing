import numpy as np

def pwmQuant(vx, sP, mode="triangle", normalize=True):
    """
    Standard PWM quantization with carrier defined by number of periods.

    Parameters
    ----------
    vx : array_like
        Input signal
    sP : int
        Number of PWM periods
    mode : str
        "triangle" or "rampup"
    normalize : bool
        Normalize vx to [-1,1]

    Returns
    -------
    vb : ndarray
        PWM output in {-1, +1}
    vc : ndarray
        Carrier signal
    sMp : int
        Samples per period
    """

    vx = np.asarray(vx, dtype=float)
    N = len(vx)

    # --- period length ---
    if N % sP != 0:
        raise ValueError("N must be divisible by sP")

    sMp = N // sP

    # --- normalize ---
    if normalize:
        max_val = np.max(np.abs(vx))
        if max_val > 0:
            vx = vx / max_val

    # --- local index ---
    n = np.arange(N)
    n_loc = n % sMp

    # --- carrier ---
    if mode == "triangle":
        vc = np.where(
            n_loc < sMp / 2,
            -1 + 4 * n_loc / sMp,
            3 - 4 * n_loc / sMp
        )

    elif mode == "rampup":
        vc = -1 + 2 * n_loc / sMp

    else:
        raise ValueError("mode must be 'triangle' or 'rampup'")

    # --- comparator ---
    vb = np.where(vx >= vc, 1.0, -1.0)

    return vb