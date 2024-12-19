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
from scipy.fftpack import hilbert

mtplt.rcParams['mathtext.fontset'] = 'stix'
mtplt.rcParams['font.family'] = 'STIXGeneral'


# %%
import sys
sys.path.append('../01_Library')
# individual packages
import sg, sa, sp, obq, filt

mtplt.close('all')

sNbins = 2048 
sK     = 64   
sBSize = 64
sHop   = 16

# %% [markdown]
# Generate input signal
sNewSignal = False
# %%
### Signal generation ###
if sNewSignal:
    sFs = 128
    sSigFmax = 23
    vxFrequ = (np.arange(0, sSigFmax, step=2)).reshape(-1, 1)
    vxPhase = np.random.rand(len(vxFrequ), 1) * 2 * np.pi

    np.save('saves/vxFrequ.npy', vxFrequ)
    np.save('saves/vxPhase.npy', vxPhase)
    np.save('saves/sSigFmax.npy', sSigFmax)
    np.save('saves/sFs.npy', sFs)
else:    
####
    vxFrequ     = np.load('saves/vxFrequ.npy')
    vxPhase     = np.load('saves/vxPhase.npy')
    sSigFmax    = np.load('saves/sSigFmax.npy')
    sFs         = np.load('saves/sFs.npy') 

sT = 1 / (sFs)
v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# %%
### Generate ideal matrices ###
vRIdeal, vW = filt.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), sMinBin=None, sType='lowpass', full=True)
mRIdeal = scLinAlg.toeplitz(vRIdeal)

sFpb        = sSigFmax
sFsb        = sFpb + 5
sApbdB      = 0.001
sAsbdB      = 40

# create Filter
(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False)
np.save('saves/vWcoeff.npy', vWcoeff)

mOnes = np.ones((sNbins,sNbins))
mSigDeltaFilt = np.tril(mOnes)

# %% [markdown]
# Quantize the input signal
nU, vW = filt.idealBinFilt(sK, sg.freq2Bin(sSigFmax, sK, sFs), sMinBin=None, sType='lowpass', full=True)

vBDFT = obq.iterBlockQDFT(vx, vW, sBSize, sK, 16)
print("Single-Iterative solution found!")
np.save('saves/vBDFT.npy', vBDFT)

vCoeffZ, mLobes = sa.detZC(vWcoeff, None)
vPruning, sPrunIdx  = sp.enLobePruning(vWcoeff, mLobes, 0.1, 8, True)
vWcoeffcut = vWcoeff[sPrunIdx+1::]

filt.plotFrequResp(vw, vH, sFs, sFpb, sFsb, sHpbMin, sHpbMax, sHsbMax, None, None, 'lowpass')
filt.anAndPlotK(vWcoeffcut, sFs, sFpb, sFsb, None, None, 'lowpass')
mtplt.tight_layout()
mtplt.pause(1)
              
#vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQnew(vx, vWcoeffcut, sBSize, 'grb')
vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQ(vx, vWcoeffcut, sBSize, 'grb')

# %%
vX = np.fft.fft(mRIdeal @ vx)
vXMag = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))

vBfft = np.fft.fft(vBDFT,sNbins)
vBfftMag = 20*sa.safelog10(np.abs(vBfft) / np.max(abs(vX))) 

vBBlockfft = np.fft.fft(vBSequBlock,sNbins)
vBBlockfftMag = 20*sa.safelog10(np.abs(vBBlockfft) / np.max(abs(vX))) 

vBReckFiltfft = np.fft.fft(vWcoeff,sNbins)
vBReckFiltfftMag = 20*sa.safelog10(np.abs(vBReckFiltfft) / np.max(abs(vBReckFiltfft))) 

vDiffBlock = vX - vBBlockfft#np.fft.fft(vWcoeffcut,sNbins)#
vDiffBlockMag = 20*sa.safelog10(np.abs(vDiffBlock) / np.max(abs(vX))) 

vBlockRec = np.fft.fft(mRIdeal @ vBSequBlock, sNbins)
vBlockRecMag = 20*sa.safelog10(np.abs(vBlockRec) / np.max(abs(vX))) 

vDiffSingle = vX - vBfft
vDiffSingleMag = 20*sa.safelog10(np.abs(vDiffSingle) / np.max(abs(vDiffSingle))) 

vSingleRec = np.fft.fft(mRIdeal @ vBDFT, sNbins)
vSingleRecMag = 20*sa.safelog10(np.abs(vSingleRec) / np.max(abs(vX))) 

# Frequency bins
vFreq = np.fft.fftfreq(sNbins, sT)


# %% [markdown]
# SNR Calculations
# Filtered Signals
vxSigFiltIdeal           = mRIdeal @ vx
vxErrFiltIdeal           = mRIdeal @ (vx-vx)
vbSequErrFiltIdeal       = mRIdeal @ (vx-vBDFT)
vbBlockSequErrFiltIdeal  = mRIdeal @ (vx-vBSequBlock)

vxSigFilt           = np.convolve(vWcoeff,vx,'same')
vxErrFilt           = np.convolve(vWcoeff,(vx-vx),'same')
vbSequErrFilt       = np.convolve(vWcoeff,(vx-vBDFT),'same')
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
pltObsOne.set_title('Difference Signal DFT')
pltObsOne.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltObsOne.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltObsOne.set_xlim([0,sFs/2])
pltObsOne.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsTwo = figOne.add_subplot(Pltgs[1,1])
pltObsTwo.plot(vFreq[:sNbins // 2], vDiffBlockMag[:sNbins // 2])
pltObsTwo.set_title('Difference Signal Block Optimization (ISCAS24)')
pltObsTwo.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltObsTwo.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltObsTwo.set_xlim([0,sFs/2])
pltObsTwo.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqOne = figOne.add_subplot(Pltgs[2,0])
pltFreqOne.plot(vFreq[:sNbins // 2], vBfftMag[:sNbins // 2])
pltFreqOne.set_title('Frequency Spectrum DFTQ SER: {snr} dB\nFrequency Spectrum DFTQ Ideal: {snr_ideal} dB'.format(snr=round(sVB_SNRdb, 2), snr_ideal=round(sVB_SNRdbIdeal, 2)))
pltFreqOne.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltFreqOne.set_ylabel('Magnitude $(dB)$', fontsize = 11)
pltFreqOne.set_xlim([0,sFs/2])
pltFreqOne.set_ylim([-60,5])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqTwo = figOne.add_subplot(Pltgs[2,1])
pltFreqTwo.plot(vFreq[:sNbins // 2], vBBlockfftMag[:sNbins // 2])
pltFreqTwo.set_title('Frequency Spectrum BOBQ(ISCAS24) SER: {snr} dB\nFrequency Spectrum BOBQ(ISCAS24) Ideal: {snr_ideal} dB'.format(snr=round(sVBBlock_SNRdb, 2), snr_ideal=round(sVBBlock_SNRdbIdeal, 2)))
pltFreqTwo.set_xlabel('Frequency $(Hz)$', fontsize = 11)
pltFreqTwo.set_ylabel('Magnitude $(dB)$', fontsize = 11)
mtplt.minorticks_on()
pltFreqTwo.set_xlim([0,sFs/2])
pltFreqTwo.set_ylim([-60,5])
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

mtplt.tight_layout(pad=0.25)
mtplt.show()

xticks = np.linspace(0, np.pi, 11)
xtick_labels = [r'$0$', r'$\frac{\pi}{10}$', r'$\frac{2\pi}{10}$', r'$\frac{3\pi}{10}$', 
                r'$\frac{4\pi}{10}$', r'$\frac{5\pi}{10}$', r'$\frac{6\pi}{10}$', 
                r'$\frac{7\pi}{10}$', r'$\frac{8\pi}{10}$', r'$\frac{9\pi}{10}$', r'$\pi$']
vNormFrequ = (vFreq / sFs) * 2* np.pi

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.75)

# NOISE Create a new figure for the additional plots (3x1)
# Convert cm to inches (1 inch = 2.54 cm)
width_cm = 12.5  # Desired width in cm
height_cm = 4  # Desired height in cm

# Convert to inches
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
figTwo = mtplt.figure(figsize=(width_inch,height_cm))
Pltgs2 = gridspec.GridSpec(2, 1)
# (2,1) Spectrum of the difference signal vDiffSingleMag with overlay of vSingleRecMag
pltCompSDQ = figTwo.add_subplot(Pltgs2[0,0])
#pltCompSDQ.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vBfftMag[:sNbins // 2], color='black', label='SDQ', linewidth=1.5)
pltCompSDQ.plot(vNormFrequ[:sNbins // 2], vBfftMag[:sNbins // 2], color='black', label='DFTQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompSDQ.set_title(r'Frequency Spectrum', fontsize=13)
pltCompSDQ.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompSDQ.set_xlim([0, np.pi])
pltCompSDQ.set_ylim([-60, 5])
txtStr = 'DFTQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2))
pltCompSDQ.text(1.75, -45, txtStr, fontsize=13, verticalalignment='top', bbox=props)
pltCompSDQ.minorticks_on()
pltCompSDQ.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
#pltCompSDQ.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompSDQ.set_xticks(xticks)
pltCompSDQ.set_xticklabels(xtick_labels, fontsize=13)

pltCompOBBQ = figTwo.add_subplot(Pltgs2[1,0])
#pltCompOBBQ.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ', linewidth=1.5)
pltCompOBBQ.plot(vNormFrequ[:sNbins // 2], vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ(ISCAS24), SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompOBBQ.set_title(r'Frequency Spectrum OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), fontsize=14)
pltCompOBBQ.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompOBBQ.set_xlim([0, np.pi])
pltCompOBBQ.set_ylim([-60, 5])
txtStr = 'OBBQ(ISCAS24), SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2))
pltCompOBBQ.text(1.75, -45, txtStr, fontsize=13, verticalalignment='top', bbox=props)
pltCompOBBQ.minorticks_on()
pltCompOBBQ.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltCompOBBQ.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltCompOBBQ.set_xlim([0,np.pi])
pltCompOBBQ.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompOBBQ.set_xticks(xticks)
pltCompOBBQ.set_xticklabels(xtick_labels, fontsize=13)
# Adjust layout
mtplt.tight_layout(pad=0.25)
mtplt.show()


width_cm = 12.5  # Desired width in cm
height_cm = 4  # Desired height in cm

# Convert to inches
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
figThree = mtplt.figure(figsize=(width_inch,height_cm))
Pltgs2 = gridspec.GridSpec(2, 1)
# (2,1) Spectrum of the difference signal vDiffSingleMag with overlay of vSingleRecMag
pltCompSDQd = figThree.add_subplot(Pltgs2[0,0])
#pltCompSDQd.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vDiffSingleMag[:sNbins // 2], color='black', label='SDQ', linewidth=1.5)
pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vDiffSingleMag[:sNbins // 2], color='black', label='DFTQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompSDQd.set_title(r'Difference Spectrum', fontsize=13)
pltCompSDQd.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompSDQd.set_xlim([0, np.pi])
pltCompSDQd.set_ylim([-60, 5])
txtStr = 'DFTQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2))
pltCompSDQd.text(1.75, -45, txtStr, fontsize=13, verticalalignment='top', bbox=props)
pltCompSDQd.minorticks_on()
pltCompSDQd.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltCompSDQd.set_xticks(xticks)
pltCompSDQd.set_xticklabels(xtick_labels, fontsize=13)

pltCompOBBQd = figThree.add_subplot(Pltgs2[1,0])
#pltCompOBBQd.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vDiffBlockMag[:sNbins // 2], color='black', label='OBBQ', linewidth=1.5)
pltCompOBBQd.plot(vNormFrequ[:sNbins // 2], vDiffBlockMag[:sNbins // 2], color='black', label='OBBQ(ISCAS24), SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompOBBQd.set_title(r'Difference Spectrum OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), fontsize=14)
pltCompOBBQd.set_ylabel(r'Magnitude (dB)', fontsize=13)

pltCompOBBQd.set_ylim([-60, 5])
txtStr = 'OBBQ(ISCAS24), SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2))
pltCompOBBQd.text(1.75, -45, txtStr, fontsize=13, verticalalignment='top', bbox=props)
pltCompOBBQd.minorticks_on()
pltCompOBBQd.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltCompOBBQd.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltCompOBBQd.set_xlim([0, np.pi])
pltCompOBBQd.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompOBBQd.set_xticks(xticks)
pltCompOBBQd.set_xticklabels(xtick_labels, fontsize=13)
# Adjust layout
mtplt.tight_layout(pad=0.25)
mtplt.show()

#####################################

