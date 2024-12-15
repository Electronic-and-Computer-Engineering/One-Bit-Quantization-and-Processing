# system packages
import numpy as np

def signalGen(v_n, vFreq, vPhaseInit, sFs, sType):
    vTime = v_n * (1 / sFs)

    if sType == 'real':
        mSig = np.sin(2 * np.pi * vFreq * vTime.T + vPhaseInit)
        vSig = np.sum(mSig, axis=0)   
    elif sType == 'delta':
        mSig = np.sinc(2 * np.pi * vFreq.T * vTime + vPhaseInit.T)
        vSig = np.sum(mSig, axis=1)
    elif sType == 'complex':
        mSig = np.exp(1j * 2 * np.pi * vFreq.T * vTime + vPhaseInit.T)
        vSig = np.sum(mSig, axis=1)
    else:
        raise ValueError('signalGen::INPUT sType is NOT supported! Try to use real or complex instead!')

    return vSig, vTime