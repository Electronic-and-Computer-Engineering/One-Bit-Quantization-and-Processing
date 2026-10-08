import numpy as np

def binaryComb(n):
    """
    Generate all possible binary combinations of length n as a matrix.
    
    Args:
    n (int): Length of the binary strings.
    
    Returns:
    numpy.ndarray: A 2D NumPy array representing all binary combinations.
    """
    sNumComb = 2 ** n
    mBinComb = np.zeros((sNumComb, n), dtype=int)
    
    for i in range(sNumComb):
        strBin = format(i, f'0{n}b')
        mBinComb[i] = [int(bit) for bit in strBin]
    
    return mBinComb