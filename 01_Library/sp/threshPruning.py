import numpy as np

def threshPruning(vCoeffs, threshold=0.01):
    """
    Perform threshold-based pruning on FIR filter coefficients and retain structure.
    
    Args:
    coefficients (numpy array): Array of FIR filter coefficients.
    threshold (float): Coefficients with absolute values below this threshold will be set to zero.
    
    Returns:
    numpy array: Pruned coefficients where insignificant values are set to zero but retain original positions.
    """
    
    # Initialize an array for pruned coefficients (same as original)
    vPrunCoeffs = np.copy(vCoeffs)
    
    # Apply threshold pruning: set coefficients with absolute value below threshold to zero
    vPrunCoeffs[np.abs(vCoeffs) < threshold] = 0
    
    vPrunCoeffs = vPrunCoeffs[vPrunCoeffs != 0]
    return vPrunCoeffs