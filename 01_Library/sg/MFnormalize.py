import numpy as np

def MFnormalize(vSignal, slwBound=-1, supBound=1):
    # Convert the signal to a numpy array for easier manipulation
    #vSignal = np.array(vSignal)
    
    # Remove the mean of the signal
    vMeanRemoved = vSignal - np.mean(vSignal)
    
    # Find the peak value (maximum of absolute values) to determine scaling factor
    sPeak = np.max(np.abs(vMeanRemoved))
    
    # Scale the signal to fit within the bounds -1 and 1
    if sPeak == 0:
        # Avoid division by zero if all values in the signal are the same
        vScSignal = vMeanRemoved
    else:
        vScSignal = vMeanRemoved / sPeak * (supBound - slwBound) / 2 + (supBound + slwBound) / 2
    
    # The clipping step is removed to preserve the dynamics of the signal
    return vScSignal