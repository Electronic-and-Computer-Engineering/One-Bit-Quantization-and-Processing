import numpy as np

def  dftMat(N, K, range=None, normalize=False, unit='rad'):
    """
    Constructs a complex-valued DFT matrix (K x N) for full or partial spectrum analysis.

    Parameters:
    ----------
    N : int
        Length of time-domain signal (number of columns).
    K : int
        Number of frequency bins (number of rows).
    range : array-like of shape (2,), optional
        Frequency interval [low, high].
        If None: uses full range [0, 2π) (in rad/sample).
    unit : str, optional
        Unit of the range: 'rad' (radians, default) or 'f' (normalized frequency in [0, 1]).
    normalize : bool, optional
        If True, applies 1/√N normalization (Parseval-compatible).

    Returns:
    -------
    F : ndarray of shape (K, N)
        Complex-valued DFT matrix.
    omega_k : ndarray of shape (K,)
        Frequency grid (in rad/sample) corresponding to the rows of F.
    """

    n = np.arange(N)  # time indices

    if range is None:
        # Full-band DFT (uniformly spaced over [0, 2π))
        omega_k = 2 * np.pi * np.arange(K) / K
    else:
        range = np.asarray(range)
        if range.shape != (2,):
            raise ValueError("range must be a 2-element array-like: [low, high]")

        if unit == 'f':
            range = 2 * np.pi * range
        elif unit != 'rad':
            raise ValueError("unit must be either 'rad' or 'f'")

        omega_k = np.linspace(range[0], range[1], K)

    F = np.exp(-1j * np.outer(omega_k, n))  # shape: (K, N)

    if normalize:
        F /= np.sqrt(N)

    return F, omega_k