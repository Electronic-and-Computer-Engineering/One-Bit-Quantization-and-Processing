import numpy as np

def convMtx(vCoeffs, sNbins, sType):
    """Generate a convolution matrix
    
    Parameters:
    vCoeffs: 1D array-like, filter kernel vector.
    sNbins: int, the number of columns in the output matrix.
    
    Returns:
    A 2D NumPy array representing the convolution matrix.
    """
    sL = len(vCoeffs)
    # Initialize the convolution matrix
    if sType == 'rowWise':
        mConvM = np.zeros((sNbins, sL+sNbins-1),'float')
    elif sType == 'colWise':
        mConvM = np.zeros((sL+sNbins-1,sNbins),'float')
    elif sType == 'rowWiseNN':
        mConvM = np.zeros((2*sL+sNbins-1, sL+sNbins-1),'float')
    elif sType == 'colWiseNN':
        mConvM = np.zeros((2*sL+sNbins-1,sL+sNbins-1),'float')
    else:
        raise ValueError('convmtx::INPUT sType is NOT supported! Try to use rowWise, colWise, rowWiseNN and, colWiseNN instead!')

    # Populate the convolution matrix
    if ((sType == 'rowWise') | (sType == 'colWise')):
        for i in range(sNbins):
            if sType == 'rowWise':
                mConvM[i, i:i+sL] = vCoeffs
            elif sType == 'colWise':
                mConvM[i:i+sL,i] = vCoeffs.T
        mConvM = mConvM[:, :sNbins]

    elif ((sType == 'rowWiseNN') | (sType == 'colWiseNN')):
        for i in range(sNbins+sL-1):
            if sType == 'rowWiseNN':
                mConvM[i, i:i+sL] = vCoeffs
            elif sType == 'colWiseNN':
                mConvM[i:i+sL,i] = vCoeffs.T
        mConvM = mConvM[:sNbins+sL-1, :sNbins+sL-1]

    return mConvM