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
import sg, sa, sp, obq, filt

mtplt.close('all')
bSNRideal = False

sNbins = 16       
sBSize = 4

#sHop = sBSize

# %% [markdown]
# Generate input signal
sNewSignal = False
# %%
### Signal generation ###
if sNewSignal:
    sFs = 16
    sSigFmax = 5
    vxFrequ = (np.arange(0, sSigFmax, step=1)).reshape(-1, 1)
    vxPhase = np.random.rand(len(vxFrequ), 1) * 2 * np.pi

    np.save('vxFrequ_small.npy', vxFrequ)
    np.save('vxPhase_small.npy', vxPhase)
    np.save('sSigFmax_small.npy', sSigFmax)
    np.save('sFs_small.npy', sFs)
else:    
####
    vxFrequ     = np.load('vxFrequ_small.npy')
    vxPhase     = np.load('vxPhase_small.npy')
    sSigFmax    = np.load('sSigFmax_small.npy')
    sFs         = np.load('sFs_small.npy') 

sT = 1 / (sFs)
v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# %%
### Generate ideal matrices ###
vRIdeal = filt.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), None, 'lowpass')
mRIdeal = scLinAlg.toeplitz(vRIdeal)

sFpb        = sSigFmax
sFsb        = sFpb + 2.75
sApbdB      = 0.00001
sAsbdB      = 23

# create Filter
#sTaps = filt.firOptNLPEqu(sFs, sFpb, sFsb, sApbdB, sAsbdB)
#(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPEqu(sFs, sFpb, sFsb, sApbdB, sAsbdB, sTaps)

(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None)
np.save('vWcoeff.npy', vWcoeff)

mOnes = np.ones((sNbins,sNbins))
mSigDeltaFilt = np.tril(mOnes)

# %% [markdown]
# Quantize the input signal
vBSequSingle, ve, ve_hat = obq.iterSequQ(vx,mSigDeltaFilt,0)
print("Single-Iterative solution found!")
np.save('vBSequSingle_small.npy', vBSequSingle)

#vWcoeff = sp.resFiltCoeffs(vWcoeff, 0.5)
vCoeffZ, tGDI, tMLobe = sa.detZC(vWcoeff, 2, None, True)
sMLobeStt, sMLobeStp, sMLobeWidth = tMLobe
sMaxGradIdx, sMaxGradVal, sMaxGrd = tGDI  
#vWcoeffcut = vWcoeff[sMaxGradIdx::] #sp.enPruning(vWcoeff, 0.97125)
#vWcoeffcut = sp.threshPruning(vWcoeff, threshold=0.009)
vWcoeffcut = vWcoeff[sMaxGradIdx::]

if (sMaxGradIdx + sBSize) > sMLobeStp:
    print("BlockSize = %d does NOT fit MainLobeWidth: %d" % (sBSize, sMLobeWidth))  
else:
    print("BlockSize = %d does fit MainLobeWidth: %d" % (sBSize, sMLobeWidth))  

filt.plotFrequResp(vw, vH, sFs, sFpb, sFsb, sHpbMin, sHpbMax, sHsbMax, None, None, 'lowpass')
#filt.anAndPlotK(vWcoeff[sMaxGradIdx:(sMaxGradIdx+sBSize)], sFs, sFpb, sFsb, 'lowpass')
mtplt.tight_layout()
mtplt.pause(1)
              
#vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQnew(vx, vWcoeffcut, sBSize, 'grb')
vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQ(vx, vWcoeffcut, sBSize, 'grb')

# %%
vX = np.fft.fft(mRIdeal @ vx)
vXMag = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))

vBfft = np.fft.fft(vBSequSingle,sNbins)
vBfftMag = 20*sa.safelog10(np.abs(vBfft) / np.max(abs(vX))) 

vBBlockfft = np.fft.fft(vBSequBlock,sNbins)
vBBlockfftMag = 20*sa.safelog10(np.abs(vBBlockfft) / np.max(abs(vX))) 

vBReckFiltfft = np.fft.fft(vWcoeff,sNbins)
vBReckFiltfftMag = 20*sa.safelog10(np.abs(vBReckFiltfft) / np.max(abs(vBReckFiltfft))) 

vDiffBlock = vX - vBBlockfft#np.fft.fft(vWcoeffcut,sNbins)#
vDiffBlockMag = 20*sa.safelog10(np.abs(vDiffBlock) / np.max(abs(vX))) 

vDiffSingle = vX - vBfft
vDiffSingleMag = 20*sa.safelog10(np.abs(vDiffSingle) / np.max(abs(vDiffSingle))) 

# Frequency bins
vFreq = np.fft.fftfreq(sNbins, sT)

# %% [markdown]
# SNR Calculations

# Filtered Signals
vxSigFiltIdeal           = mRIdeal @ vx
vxErrFiltIdeal           = mRIdeal @ (vx - vx)
vbSequErrFiltIdeal       = mRIdeal @ (vx-vBSequSingle)
vbBlockSequErrFiltIdeal  = mRIdeal @ (vx-vBSequBlock)

vxSigFilt           = np.convolve(vWcoeff,vx,'same')
vxErrFilt           = np.convolve(vWcoeff,(vx-vx),'same')
vbSequErrFilt       = np.convolve(vWcoeff,(vx-vBSequSingle),'same')
vbBlockSequErrFilt  = np.convolve(vWcoeff,(vx-vBSequBlock),'same')

sVX_MSEIdeal, sVX_SNRdbIdeal, sVX_PSNRdbIdeal = sa.evalN(vxErrFiltIdeal, vxSigFiltIdeal)
sVB_MSEIdeal, sVB_SNRdbIdeal, sVB_PSNRdbIdeal = sa.evalN(vbSequErrFiltIdeal, vxSigFiltIdeal)
sVBBlock_MSEIdeal, sVBBlock_SNRdbIdeal, sVBBlock_PSNRdbIdeal = sa.evalN(vbBlockSequErrFiltIdeal, vxSigFiltIdeal)    
    
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
pltDiscTime.set_title('Input Signal SER: {snr} dB\nInput Signal SER Ideal: {snr_ideal} dB'.format(snr=round(sVX_SNRdb, 2), snr_ideal=round(sVX_SNRdbIdeal, 2)))
pltDiscTime.set_xlabel('Samples $n$', fontsize = 11)
pltDiscTime.set_ylabel('Amplitude', fontsize = 11)
pltDiscTime.set_xlim([0,sNbins])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsOne = figOne.add_subplot(Pltgs[1,0])
pltObsOne.plot(vFreq[:sNbins // 2], vDiffSingleMag[:sNbins // 2])
#pltObsOne.plot(sFiltst, vWLs[sFiltst], 'rx', markersize=6, markeredgewidth=2)
pltObsOne.set_title('Difference Signal SingleSequ')
pltObsOne.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltObsOne.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltObsOne.set_xlim([0,sFs/2])
pltObsOne.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsTwo = figOne.add_subplot(Pltgs[1,1])
pltObsTwo.plot(vFreq[:sNbins // 2], vDiffBlockMag[:sNbins // 2])
pltObsTwo.set_title('Difference Signal Block Optimization')
pltObsTwo.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltObsTwo.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltObsTwo.set_xlim([0,sFs/2])
pltObsTwo.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqOne = figOne.add_subplot(Pltgs[2,0])
pltFreqOne.plot(vFreq[:sNbins // 2], vBfftMag[:sNbins // 2])
pltFreqOne.set_title('Frequency Spectrum SDQ SER: {snr} dB\nFrequency Spectrum SDQ Ideal: {snr_ideal} dB'.format(snr=round(sVB_SNRdb, 2), snr_ideal=round(sVB_SNRdbIdeal, 2)))
pltFreqOne.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltFreqOne.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltFreqOne.set_xlim([0,sFs/2])
pltFreqOne.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqTwo = figOne.add_subplot(Pltgs[2,1])
pltFreqTwo.plot(vFreq[:sNbins // 2], vBBlockfftMag[:sNbins // 2])
pltFreqTwo.set_title('Frequency Spectrum BOBQ SER: {snr} dB\nFrequency Spectrum BOBQ Ideal: {snr_ideal} dB'.format(snr=round(sVBBlock_SNRdb, 2), snr_ideal=round(sVBBlock_SNRdbIdeal, 2)))
pltFreqTwo.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltFreqTwo.set_ylabel('Magnitude $(dB)$', fontsize = 11)
mtplt.minorticks_on()
pltFreqTwo.set_xlim([0,sFs/2])
pltFreqTwo.set_ylim([-60,5])
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

mtplt.tight_layout(pad=0.25)
mtplt.show()


