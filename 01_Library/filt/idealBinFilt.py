import numpy as np
import matplotlib.pyplot as plt

def idealBinFilt(sNbins, sMaxBin, sMinBin=None, sType='lowpass', full=False):
    """
    Create ideal lowpass, highpass, or bandpass filter coefficients and spectrum.

    Parameters:
    sNbins : int
        Number of bins (samples) in the filter.
    sMaxBin : int
        Maximum bin (frequency) to pass (for lowpass/bandpass) or stop (for highpass).
    sMinBin : int, optional
        Minimum bin (frequency) to pass (only used for bandpass).
    sType : str
        Type of filter: 'lowpass', 'highpass', 'bandpass'.
    full : bool, optional
        If True, return the full spectrum (positive and negative frequencies).
        If False (default), return only the positive frequencies up to f_s/2.

    Returns:
    vFiltCoeffs : numpy.ndarray
        The time-domain filter coefficients.
    vSpectrum : numpy.ndarray
        The frequency-domain spectrum of the filter.
    """
    # Generate the bin indices
    v_n = np.arange(sNbins)
    
    # Create the ideal filter based on the type
    if sType == 'lowpass':
        # Lowpass filter
        vIdealFilter = np.zeros(sNbins)
        vIdealFilter[:sMaxBin] = 1
        vIdealFilter[-sMaxBin:] = 1
    
    elif sType == 'highpass':
        # Highpass filter
        vIdealFilter = np.ones(sNbins)
        vIdealFilter[:sMaxBin] = 0
        vIdealFilter[-sMaxBin:] = 0
    
    elif sType == 'bandpass':
        if sMinBin is None:
            raise ValueError("sMinBin must be provided for bandpass filter.")
        
        # Bandpass filter
        vIdealFilter = np.zeros(sNbins)        
        # Handle positive frequencies
        vIdealFilter[sMinBin:sMaxBin] = 1
        # Mirror around sNbins/2 for negative frequencies
        vIdealFilter[sNbins-sMaxBin:sNbins-sMinBin+1] = 1
        
    else:
        raise ValueError("Unsupported filter type. Use 'lowpass', 'highpass', or 'bandpass'.")

    # Perform IFFT to get filter coefficients in time domain
    vFiltCoeffs = np.fft.ifft(vIdealFilter)
    
    # Handle the full or half spectrum output
    if not full:
        # Return only the first half of the spectrum (positive frequencies)
        vSpectrum = vIdealFilter[:sNbins // 2 + 1]
    else:
        # Return the full spectrum (positive and negative frequencies)
        vSpectrum = vIdealFilter

    # Return both the time-domain coefficients and spectrum
    return vFiltCoeffs.real, vSpectrum