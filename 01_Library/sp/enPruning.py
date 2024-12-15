import numpy as np

def enPruning(vCoeffs, ePerc=0.95):
    """
    Perform energy-based pruning on FIR filter coefficients and retain structure.
    
    Args:
    coefficients (numpy array): Array of FIR filter coefficients.
    energy_threshold_percentage (float): Percentage of total energy to retain (default is 0.95, i.e., 95%).
    
    Returns:
    numpy array: Pruned coefficients where insignificant values are set to zero but retain original positions.
    """
    
    # Compute total energy of the filter coefficients
    sTotEn = np.sum(vCoeffs**2)
    
    # Define the actual energy threshold based on the percentage
    sEtH = ePerc * sTotEn
    
    # Sort coefficients by their contribution to energy (absolute squared values)
    vSortIdx = np.argsort(vCoeffs**2)[::-1]  # Sort descending by energy contribution
    
    # Initialize an array for pruned coefficients with zeros (same length as the original)
    vPrunCoeffs = np.zeros_like(vCoeffs)
    
    # Prune coefficients: accumulate energy and retain only those contributing to the threshold percentage
    sCumEn = 0
    for idx in vSortIdx:
        sCumEn += vCoeffs[idx]**2
        if sCumEn >= sEtH:
            break
        vPrunCoeffs[idx] = vCoeffs[idx]  # Keep significant coefficients in original position
    
    #vPrunCoeffs = vPrunCoeffs[vPrunCoeffs != 0]
    return vPrunCoeffs