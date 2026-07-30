import numpy as np

import numpy as np
from scipy.interpolate import interp1d

def symEnergyWin(w, sM=None, energy_percent=0.95):
    """
    Extracts the central, symmetric portion of a window containing at least 
    `energy_percent` of the total energy, then interpolates it to length sM.
    
    Parameters:
        w               : 1D numpy array (window function)
        sM              : desired output length (defaults to len(w))
        energy_percent  : float (e.g. 0.95 for 95% energy)

    Returns:
        w_interp        : interpolated symmetric window of length sM
        i_start, i_end  : original index range selected
    """
    if sM is None:
        sM = len(w)
    
    e = w ** 2
    c = np.cumsum(e)
    total_energy = c[-1]
    target = energy_percent * total_energy
    
    N = len(w)
    center = N // 2
    best_half_len = N
    i_start, i_end = 0, N - 1

    # Expand symmetric window from center outward
    for half_len in range(1, center + 1):
        i0 = center - half_len
        i1 = center + half_len
        if i0 < 0 or i1 > N:
            continue
        energy = np.sum(e[i0:i1])
        if energy >= target:
            best_half_len = half_len
            i_start = i0
            i_end = i1
            break

    # Cut and interpolate
    w_cut = w[i_start:i_end]
    x_old = np.linspace(0, 1, num=len(w_cut))
    x_new = np.linspace(0, 1, num=sM)

    interp_func = interp1d(x_old, w_cut, kind='linear')
    w_interp = interp_func(x_new)

    return w_interp, i_start, i_end