# %%
# system packages
import numpy as np
#import pandas as pd
#import math

#Plotting
import matplotlib.pyplot as mtplt
import matplotlib.gridspec as gridspec

#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import scipy.signal as sigP

# %%
# individual packages
import sg, sa, sp, obq
mtplt.close('all')
bSNRideal = False

sNbins = 2**12
sFs = 4096
sT = 1 / (sFs)
       
sL = 60
sBSize = 64
sPadLen = sNbins + (sL - 1)
sSigFmax = 63 


# %% [markdown]
# Generate input signal

# %%
### Signal generation ###
v_n = np.arange(sNbins).reshape(-1, 1)
vxFrequ = (np.arange(0, sSigFmax, step=2)).reshape(-1, 1)
vxPhaseInit = np.random.rand(len(vxFrequ), 1) * 2 * np.pi

vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhaseInit, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# Save to a file
#np.save('signal.npy', vx)
#
vx = np.load('signal.npy')

# %% [markdown]
# Generate FiltersCoeff and corresp. Matrices

# %%
### Generate ideal matrices ###
vRIdeal = sp.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), 'normal')
mRIdeal = scLinAlg.toeplitz(vRIdeal)

### Filter Design ###
#The desired width of the transition from pass to stop,
# relative to the Nyquist rate.
sTransWidthHz = 225 #109
sTransWidthDig = sTransWidthHz / (sFs /2)

# The desired attenuation in the stop band, in dB.
sAttdB = 40.0 #29

# Compute the order and Kaiser parameter for the FIR filter.
sTaps, sBeta = sigP.kaiserord(sAttdB, sTransWidthDig)

# The cutoff frequency of the filter.
sCutOffHz = sSigFmax + 50  # +5        #Hz
sCutOffDig = sCutOffHz / (sFs /2)

# Use firwin with a Kaiser window to create a lowpass FIR filter.
vWLs = sigP.firwin(sTaps, sCutOffDig, window=('kaiser', sBeta))

vNormWLs = vWLs / np.sum(vWLs)

mOnes = np.ones((sNbins,sNbins))
mSigDeltaFilt = np.tril(mOnes)

# %% [markdown]
# Quantize the input signal

# %%
vBSequSingle, ve, ve_hat = obq.iterSequQ(vx,mSigDeltaFilt,0)
print("Single-Iterative solution found!")
#vBSequBlock, vEBlock = obq.fullOpt(vx,mW[sCut:sCut+len(vx)],vBSequSingle)
vW = vNormWLs[0::]
vBSequBlock, vEBlock = obq.iterBlockQ(vx, vW, sBSize, 'grb')
#vBSequBlock = vBSequBlock[sPadAdd::]

# %% [markdown]
# Frequency Analysis

# %%
vX = np.fft.fft(mRIdeal @ vx)
vXMag = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))

vBfft = np.fft.fft(vBSequSingle,sNbins)
vBfftMag = 20*sa.safelog10(np.abs(vBfft) / np.max(abs(vX))) 

vBBlockfft = np.fft.fft(vBSequBlock,sNbins)
vBBlockfftMag = 20*sa.safelog10(np.abs(vBBlockfft) / np.max(abs(vX))) 

vBReckFiltfft = np.fft.fft(vW,sNbins)
vBReckFiltfftMag = 20*sa.safelog10(np.abs(vBReckFiltfft) / np.max(abs(vBReckFiltfft))) 

vDiff = vX - vBBlockfft
vDiffMag = 20*sa.safelog10(np.abs(vDiff) / np.max(abs(vX))) 


# Frequency bins
vFreq = np.fft.fftfreq(sNbins, sT)

# %% [markdown]
# SNR Calculations

# Filtered Signals
if bSNRideal: 
    vxSigFilt           = mRIdeal @ vx
    vxErrFilt           = mRIdeal @ (vx - vx)
    vbSequErrFilt       = mRIdeal @ (vx-vBSequSingle)
    vbBlockSequErrFilt  = mRIdeal @ (vx-vBSequBlock)
else:
    vxSigFilt           = np.convolve(vW,vx,'same')
    vxErrFilt           = np.convolve(vW,(vx-vx),'same')
    vbSequErrFilt       = np.convolve(vW,(vx-vBSequSingle),'same')
    vbBlockSequErrFilt  = np.convolve(vW,(vx-vBSequBlock),'same')
    
    
sVX_MSE, sVX_SNRdb, sVX_PSNRdb = sa.evalN(vxErrFilt, vxSigFilt)
sVB_MSE, sVB_SNRdb, sVB_PSNRdb = sa.evalN(vbSequErrFilt, vxSigFilt)
sVBBlock_MSE, sVBBlock_SNRdb, sVBBlock_PSNRdb = sa.evalN(vbBlockSequErrFilt, vxSigFilt)

# %% [markdown]
# Plots

# %%
###### PLOTTING ######
figOne = mtplt.figure()
Pltgs = gridspec.GridSpec(3, 2)

pltDiscTime = figOne.add_subplot(Pltgs[0,:])
pltDiscTime.plot(vx)
pltDiscTime.set_title('Input Signal SER: {snr} dB'.format(snr = round(sVX_SNRdb,2)))
pltDiscTime.set_xlabel('Samples $n$', fontsize = 11)
pltDiscTime.set_ylabel('Amplitude', fontsize = 11)
pltDiscTime.set_xlim([0,sNbins])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsOne = figOne.add_subplot(Pltgs[1,0])
pltObsOne.plot(vFreq[:sNbins // 2], vDiffMag[:sNbins // 2])
#pltObsOne.plot(sFiltst, vWLs[sFiltst], 'rx', markersize=6, markeredgewidth=2)
pltObsOne.set_title('')
pltObsOne.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltObsOne.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltObsOne.set_xlim([0,sFs/2])
pltObsOne.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsTwo = figOne.add_subplot(Pltgs[1,1])
pltObsTwo.plot(vFreq[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2])
pltObsTwo.set_title('Frequency Response of $W$')
pltObsTwo.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltObsTwo.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltObsTwo.set_xlim([0,sFs/2])
pltObsTwo.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqOne = figOne.add_subplot(Pltgs[2,0])
pltFreqOne.plot(vFreq[:sNbins // 2], vBfftMag[:sNbins // 2])
pltFreqOne.set_title('Frequency Spectrum SDQ SER: {snr} dB'.format(snr = round(sVB_SNRdb,2)))
pltFreqOne.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltFreqOne.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltFreqOne.set_xlim([0,sFs/2])
pltFreqOne.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqTwo = figOne.add_subplot(Pltgs[2,1])
pltFreqTwo.plot(vFreq[:sNbins // 2], vBBlockfftMag[:sNbins // 2])
pltFreqTwo.set_title('Frequency Spectrum BOBQ SER: {snr} dB'.format(snr = round(sVBBlock_SNRdb,2)))
pltFreqTwo.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltFreqTwo.set_ylabel('Magnitude $(dB)$', fontsize = 11)
mtplt.minorticks_on()
pltFreqTwo.set_xlim([0,sFs/2])
pltFreqTwo.set_ylim([-60,5])
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

mtplt.tight_layout(pad=0.25)
mtplt.show()


