#sympi
import sympy as sypy

def convmtx(vCoeffs, sNbins, sType):
    """Generate a convolution matrix using SymPy for symbolic computations.
    
    Parameters:
    vCoeffs: list or Matrix of symbols, filter kernel vector.
    sNbins: int, the number of columns in the output matrix.
    
    Returns:
    A SymPy Matrix representing the convolution matrix.
    """
    sL = len(vCoeffs)
    vCoeffs = sypy.Matrix(vCoeffs)  # Ensure vCoeffs is a SymPy Matrix
    # Initialize the convolution matrix
    if sType == 'rowWise':
        mConvM = sypy.zeros(sNbins, sL + sNbins - 1)
    elif sType == 'colWise':
        mConvM = sypy.zeros(sL + sNbins - 1, sNbins)
    elif sType == 'rowWiseNN':
        mConvM = sypy.zeros(2 * sL + sNbins - 1, sL + sNbins - 1)
    elif sType == 'colWiseNN':
        mConvM = sypy.zeros(2 * sL + sNbins - 1, sL + sNbins - 1)
    else:
        raise ValueError('convmtx::INPUT sType is NOT supported! Try to use rowWise, colWise, rowWiseNN, and colWiseNN instead!')

    # Populate the convolution matrix
    if sType == 'rowWise' or sType == 'colWise':
        for i in range(sNbins):
            if sType == 'rowWise':
                mConvM[i, i:i+sL] = vCoeffs
            elif sType == 'colWise':
                for j in range(sL):
                    mConvM[i+j, i] = vCoeffs[j]
        if sType in ['rowWise', 'colWise']:
            mConvM = mConvM[:, :sNbins]

    elif sType == 'rowWiseNN' or sType == 'colWiseNN':
        for i in range(sNbins + sL - 1):
            if sType == 'rowWiseNN':
                mConvM[i, i:i+sL] = vCoeffs
            elif sType == 'colWiseNN':
                for j in range(sL):
                    mConvM[i+j, i] = vCoeffs[j]
        if sType in ['rowWiseNN', 'colWiseNN']:
            mConvM = mConvM[:sNbins + sL - 1, :sNbins + sL - 1]

    return mConvM