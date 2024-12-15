import numpy as np

def gdShift(vSig, sGd):
   sN = len(vSig)
   vFreq = np.fft.fftfreq(sN)
   vSigFFT = np.fft.fft(vSig)
   vPhaseShift = np.exp(-1j * 2 * np.pi * vFreq * sGd)
   vShiftedFFT = vSigFFT * vPhaseShift
   vSigShifted = np.fft.ifft(vShiftedFFT)
   
   return np.real(vSigShifted)