import numpy as np
from scipy.signal import get_window

def dftMat(N, K, freq_range=None, omega_k=None, normalize=False, unit='rad', window=None):
    """
    Constructs a complex-valued DFT matrix (K x N), optionally windowed.

    Parameters:
    ----------
    N : int
        Number of time-domain samples (columns).
    K : int
        Number of frequency bins (rows).
    freq_range : array-like of shape (2,), optional
        Frequency interval [low, high]. If None, uses full range [0, 2π).
    omega_k : ndarray, optional
        Custom frequency grid in radians/sample.
    normalize : bool, optional
        If True, applies 1/√N normalization.
    unit : {'rad', 'f'}, optional
        Unit for freq_range boundaries ('rad' or 'f').
    window : str or array-like or None
        Optional window function. If string, must be valid for `scipy.signal.get_window`.

    Returns:
    -------
    F : ndarray (K x N)
        Complex-valued DFT matrix (optionally windowed).
    omega_k : ndarray (K,)
        Frequency grid in radians/sample.
    """
    n = np.arange(N)

    # Frequency axis
    if omega_k is not None:
        omega_k = np.asarray(omega_k)
        K = len(omega_k)
    elif freq_range is not None:
        freq_range = np.asarray(freq_range)
        if freq_range.shape[0] < 2:
            raise ValueError("freq_range must contain at least two elements")

        if unit == 'f':
            freq_range = 2 * np.pi * freq_range
        elif unit != 'rad':
            raise ValueError("unit must be either 'rad' or 'f'")

        omega_k = np.linspace(freq_range[0], freq_range[-1], K, endpoint=False)
    else:
        if K is None:
            raise ValueError("Either 'K' or 'omega_k' must be specified.")
        omega_k = 2 * np.pi * np.arange(K) / K

    # Base complex exponentials
    F = np.exp(-1j * np.outer(omega_k, n))  # shape (K, N)

    # Apply optional window
    if window is not None:
        if isinstance(window, str):
            w = get_window(window, N, fftbins=False)
        else:
            w = np.asarray(window)
            if w.shape != (N,):
                raise ValueError("window must have shape (N,)")

        w = w / np.linalg.norm(w)
        F = F * w[None, :]  # Broadcast multiplication

    if normalize:
        F /= N

    return F, omega_k