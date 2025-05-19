import numpy as np
import sg

def idealBinFilt(sNbins, mRanges, vMaxVal, sMinVal, bFull=False):
    """
    Create ideal lowpass, highpass, or bandpass filter coefficients and spectrum.

    Parameters:
    sNbins : int
        Number of bins (samples) in the filter.
    mRanges: int
        matrix of ranges in bin (m x 2):
            m filter ranges with [sMinRad,sMaxRad] each
    vMaxVal: int
        matrix of [sMinVal, sMaxVal] for each m filter (m x 2)
    sMinVal: int
        minimum value for all NON ranged values    
    full : bool, optional
        If True, return the full spectrum (positive and negative frequencies).
        If False (default), return only the positive frequencies up to pi.

    Returns:
    vFiltCoeffs : numpy.ndarray
        The time-domain filter coefficients.
    vSpectrum : numpy.ndarray
        The frequency-domain spectrum of the filter.
    """
    
    vIdealSpectrum = np.ones(sNbins) * sMinVal
    
    if mRanges.ndim == 1:
        sRows = 1
        if mRanges[0] < 0:
            raise ValueError(f"Unsupported minimum value: Value smaller than 0!")
        elif mRanges[1] > np.pi:
            raise ValueError(f"Unsupported maximum value: Value bigger than pi!")    
        
        sMinBin = sg.rad2bin(mRanges[0],sNbins)
        sMaxBin = sg.rad2bin(mRanges[1],sNbins)

        vIdealSpectrum[sMinBin:sMaxBin] = vMaxVal
        
    elif mRanges.ndim == 2:
        sRows = mRanges.shape[0] 
        for i in range(sRows):
            if mRanges[i,0] < 0:
                raise ValueError(f"Unsupported minimum value in range {i}: Value smaller than 0!")
            elif mRanges[i,1] > np.pi:
                raise ValueError(f"Unsupported maximum value in range {i}: Value bigger than pi!")    
            
            sMinBin = sg.rad2bin(mRanges[i,0],sNbins)
            sMaxBin = sg.rad2bin(mRanges[i,1],sNbins)
    
            vIdealSpectrum[sMinBin:sMaxBin] = vMaxVal[i]
        
    if bFull == True:
        vUpperHalf = vIdealSpectrum[0:sNbins//2].copy()
        vUpperHalf = vUpperHalf[::-1]
        vIdealSpectrum[sNbins//2::] = vUpperHalf.copy()
        
    # Perform IFFT to get filter coefficients in time domain
    vFiltCoeffs      = np.fft.ifft(vIdealSpectrum)
    vFiltCoeffsShift = np.fft.ifftshift(vFiltCoeffs)
    
    # Return both the time-domain coefficients and spectrum
    return vFiltCoeffs.real, vFiltCoeffsShift.real, vIdealSpectrum