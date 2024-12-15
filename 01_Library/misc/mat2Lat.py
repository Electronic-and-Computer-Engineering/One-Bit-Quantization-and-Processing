import numpy as np

def mat2Lat(vmVecMat, sDec):
    """
    Generates LaTeX code for a given numpy array with specified decimal rounding.

    Parameters:
        vector_or_matrix (np.array): The numpy array (vector or matrix).
        decimals (int): The number of decimal places to round to.

    Returns:
        str: A string containing LaTeX code for the matrix.
    """
    # Handle the rounding
    mRoundedVals = np.round(vmVecMat, sDec)

    # Determine if it's a vector or a matrix
    if mRoundedVals.ndim == 1:
        strLat = "\\begin{pmatrix} "
        strLat += ' \\ '.join(f'{num}' for num in mRoundedVals)
        strLat += ' \end{bmatrix}'
    else:
        strLat = "\\begin{bmatrix} "
        strRows = [' & '.join(f'{num}' for num in row) for row in mRoundedVals]
        strLat += ' \\ '.join(strRows)
        strLat += ' \end{bmatrix}'

    return strLat