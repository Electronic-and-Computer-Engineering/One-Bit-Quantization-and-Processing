import numpy as np

def detZC(vVec, vXvals=None):
    """
    Detects zero-crossings and identifies lobes in a 1D vector.
    
    Parameters:
        vVec (list or np.array): The input vector.
        vXvals (list or np.array, optional): Optional x coordinates corresponding to the vector elements.
                                             If None, indices of the vector are used as x coordinates.

    Returns:
        tuple: 
               - np.array or None: Exact zero-crossing points (None if no zero crossings are detected).
               - list of tuples: Lobes (start_idx, stop_idx, width) based on integer indices, or None if no zero crossings.
    """
    # Ensure the input is a numpy array
    vVec = np.array(vVec)
    
    if vXvals is None:
        vXvals = np.arange(len(vVec))
    else:
        vXvals = np.array(vXvals)

    # Detect zero crossings by checking where the sign changes
    vSigns = np.sign(vVec)
    vZC = np.where(np.diff(vSigns))[0]

    # Initialize arrays for storing zero-crossing points
    vZCRes = []
    
    # Linear interpolation to find the exact zero-crossing points
    for idx in vZC:
        x1, x2 = vXvals[idx], vXvals[idx + 1]
        y1, y2 = vVec[idx], vVec[idx + 1]
        
        # Calculate the exact x-coordinate of the zero crossing
        sXZero = x1 - y1 * (x2 - x1) / (y2 - y1)
        vZCRes.append(sXZero)

    vZCRes = np.array(vZCRes)
    
    # Handle the case where no zero crossings are detected
    if len(vZCRes) == 0:
        # Define a single "main lobe" from the start to the end of the vector
        tLobes = [(0, len(vVec)-1, len(vVec))]  # Single lobe covering the entire vector
        vZCRes = None  # No zero crossings to report
    else:
        # Define lobes based on integer indices between zero crossings
        tLobes = []
        for i in range(len(vZCRes) - 1):
            start_idx = np.searchsorted(vXvals, vZCRes[i], side='right')  # First index after the zero crossing
            stop_idx = np.searchsorted(vXvals, vZCRes[i + 1], side='left') - 1  # Last index before the next zero crossing
            width = stop_idx - start_idx + 1  # Ensure width is inclusive of the lobe range
            tLobes.append((start_idx, stop_idx, width))
    
    return vZCRes, tLobes