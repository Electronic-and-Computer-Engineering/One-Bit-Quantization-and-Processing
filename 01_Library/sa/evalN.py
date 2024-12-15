import numpy as np
import math

def evalN(vError, vRef):
    """
    Compute the SNR of a signal and a reference signal.

    Inputs:
        vError: error signal
        vRef: reference signal

    Outputs:
        sMSE: Mean Squared Error
        sSERdB: resulting SNR in dB
        sPSERdB: resulting peakSNR in dB

    Authors:
        Florian Mayer <florian.mayer@fh-joanneum.at>

    """
    vError += math.ulp(1.00)
    vRef += math.ulp(1.00)

    # Mean-Squared Error
    #sMSE = np.sum((vError) ** 2) / np.size(vRef)
    #varRef = np.sum(np.abs(vRef) ** 2) / np.size(vRef)
    #varError = np.sum(np.abs(vError) ** 2) / np.size(vError)
    sMSE = np.sum((vError) ** 2)
    varRef = np.sum(np.abs(vRef) ** 2)
    varError = np.sum(np.abs(vError) ** 2)

    # Signal-to-Noise Ratio in dB
    sSERdB = 10 * np.log10(varRef / varError)

    # Peak Signal-to-Noise Ratio in dB
    sPSERdB = 10 * np.log10(np.max(vRef) ** 2 / sMSE)

    return sMSE, sSERdB, sPSERdB