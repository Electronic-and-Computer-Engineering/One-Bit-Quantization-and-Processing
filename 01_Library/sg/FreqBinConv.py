def freq2Bin(sFreq, sNbins, sFs):
    """
    Calculates the bin number for a given frequency.

    Parameters:
    - sFreq : float
        The frequency to convert, in Hz.
    - sNbins : int
        The total number of bins (samples) in the FFT.
    - sFs : float
        The sampling frequency of the signal, in Hz.

    Returns:
    - sBin : int
        The bin number corresponding to the given frequency.
    """
    
    sBin = round(sFreq * (sNbins / sFs))
    return sBin

def bin2Freq(sBin, sNbins, sFs):
    """
    Calculates the bin number for a given frequency.

    Parameters:
    - sBin : int
        The bin number corresponding to the desired frequency.
    - sNbins : int
        The total number of bins (samples) in the FFT.
    - sFs : float
        The sampling frequency of the signal, in Hz.

    Returns:
    - sFreq : float
        The frequency converted, in Hz.
    """
    
    sFreq = round(sBin * (sFs / sNbins))
    return sFreq

