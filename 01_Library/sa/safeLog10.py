# system packages
import numpy as np

def safelog10(mInput, cDB_min=-350):
    """
    Safely computes the logarithm base 10 of the input array.
    
    Parameters:
    - mInput: numpy array, the input signal.
    - cDB_min: float, minimum dB value for replacing zeros in the input array.
    
    Returns:
    - mLog10: numpy array, the logarithm base 10 of the input array, with zeros replaced by a minimum value.
    """
    cMin = 10**(cDB_min/20)
    # Find zero elements and replace them with cMin
    mZeroElements = np.where(mInput == 0)
    if len(mZeroElements[0]) > 0:  # Check if there are any zero elements
        mInput[mZeroElements] = cMin
    mLog10 = np.log10(mInput)
    
    return mLog10