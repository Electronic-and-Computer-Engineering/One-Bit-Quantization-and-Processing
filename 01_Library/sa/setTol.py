import numpy as np

def setTol(vx, sTol=1e-12, sValRe=0.0, sValIm=0.0, strFlag='Set'):
    """
    Sets the real and imaginary parts of an array independently
    to specified values if their absolute magnitude is below a given tolerance.

    Parameters
    ----------
    vx : np.ndarray
        Input array (real or complex).
    sTol : float
        Tolerance threshold. Values smaller than this will be replaced.
    sValRe : float
        Replacement value for small real parts.
    sValIm : float
        Replacement value for small imaginary parts.
    sFlag : String
        'Set'    : Set variable to a defined value
        'Bypass' : ByBypassing for tests or comparisons
        'Snap'   : Snap variable to the next value

    Returns
    -------
    np.ndarray
        Array of the same shape, with small real/imag components replaced.
    """
    
    vx = np.asarray(vx, dtype=np.complex128)
    vRe = np.real(vx)
    vIm = np.imag(vx)
    # Skip processing if bypass flag is set
    if (strFlag == 'Bypass'):
        return vx
    
    elif (strFlag == 'Set'):
        mask_re = np.abs(vRe) < sTol
        mask_im = np.abs(vIm) < sTol
        vRe[mask_re] = sValRe
        vIm[mask_im] = sValIm
                
    elif (strFlag == 'Snap'):       
        # Compute nearest integers
        vNearRe = np.round(vRe)
        vNearIm = np.round(vIm)
        # Create masks for values close enough to their nearest integer
        mask_re = np.abs(vRe - vNearRe) < sTol
        mask_im = np.abs(vIm - vNearIm) < sTol   
        # Snap those values
        vRe[mask_re] = vNearRe[mask_re]
        vIm[mask_im] = vNearIm[mask_im]
    
    return vRe + 1j * vIm