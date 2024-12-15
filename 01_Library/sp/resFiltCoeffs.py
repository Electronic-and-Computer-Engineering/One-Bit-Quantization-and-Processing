import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

def resFiltCoeffs(vFircoeffs, sResFac):
    """
    Resample FIR filter coefficients by a given factor.

    Parameters:
    - fir_coeffs: array-like, the original FIR filter coefficients.
    - resample_factor: float, the factor by which to resample the coefficients. 
                       (e.g., 2.0 for upsampling by 2, 0.5 for downsampling by 2)

    Returns:
    - resampled_coeffs: array-like, the resampled FIR filter coefficients.
    """
    # Determine the number of coefficients in the resampled filter
    sNumCoeffs = int(np.ceil(len(vFircoeffs) * sResFac))

    # Resample the FIR coefficients using the scipy resample function
    vResCoeffs = resample(vFircoeffs, sNumCoeffs)
    
    return vResCoeffs