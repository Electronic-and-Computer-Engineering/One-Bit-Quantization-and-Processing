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

sNbins = 1024    
sBSize = 32

#sHop = sBSize

# %% [markdown]
# Generate input signal
sNewSignal = False
# %%
### Signal generation ###
if sNewSignal:
    sFs = 1024
    sSigFmax = 41
    vxFrequ = (np.arange(0, sSigFmax, step=1)).reshape(-1, 1)
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
vLPRangeFs          = [0, 0, sSigFmax, sSigFmax]
vLPfilterRad        = sg.freq2rad(vLPRangeFs, sFs)
vRIdeal, vRIdealShift, vW = filt.idealBinFilt(sNbins, vLPfilterRad, 1.0, 0.0, True)
#vRIdeal, vW = filt.idealBinFilt(sNbins, sg.freq2bin(sSigFmax, sNbins, sFs), sMinBin=None, sType='lowpass', full=True)
mRIdeal = scLinAlg.toeplitz(vRIdeal)

sFpb        = sSigFmax
sFsb        = sFpb + 21
sApbdB      = 0.01
sAsbdB      = 32

# create Filter
(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False)
np.save('vWcoeff.npy', vWcoeff)

mOnes = np.ones((sNbins,sNbins))
mSigDeltaFilt = np.tril(mOnes)

vCoeffZ, mLobes = sa.detZC(vWcoeff, None)
vPruning, sPrunIdx  = sp.enLobePruning(vWcoeff, mLobes, 0.2, 8, True)
vWcoeffcut = vWcoeff[sPrunIdx::]

filt.plotFrequResp(vw, vH, sFs, sFpb, sFsb, sHpbMin, sHpbMax, sHsbMax, None, None, 'lowpass')
filt.anAndPlotK(vWcoeffcut, sFs, sFpb, sFsb, None, None, 'lowpass')
mtplt.tight_layout()
mtplt.pause(1)

# %% [markdown]
# Quantize the input signal (SDQ)
vBSDQ, ve, ve_hat = obq.iterSequQ(vx,mSigDeltaFilt,0)
# Quantize the input signal (OBAQ)
sL = len(vWcoeff)
mOBAQ = sp.convMtx(vWcoeffcut, sNbins, 'colWise')
vBSequSingle, ve, ve_hat = obq.iterSequQ(vx,mOBAQ,0)
print("Single-Iterative solution found!")
np.save('vBSequSingle.npy', vBSequSingle)
              
#vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQnew(vx, vWcoeffcut, sBSize, 'grb')
vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQ_OA(vx, vWcoeff, sBSize, 'grb')

## %%
## Analysis ##
with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
    vX = np.fft.fft(mRIdeal @ vx)
    vxSigFiltIdeal           = mRIdeal @ vx
    vxErrFiltIdeal           = mRIdeal @ (vx-vx)
    vbSDQErrFiltIdeal        = mRIdeal @ (vx-vBSDQ)
    vbSequErrFiltIdeal       = mRIdeal @ (vx-vBSequSingle)
    vbBlockSequErrFiltIdeal  = mRIdeal @ (vx-vBSequBlock)

# %%
vX_org = np.fft.fft(vx)
vXMag_org = 20*sa.safelog10(np.abs(vX_org) / np.max(abs(vX_org)))


vXMag = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))

vBSDQfft = np.fft.fft(vBSDQ,sNbins)
vBSDQfftMag = 20*sa.safelog10(np.abs(vBSDQfft) / np.max(abs(vX))) 

vBfft = np.fft.fft(vBSequSingle,sNbins)
vBfftMag = 20*sa.safelog10(np.abs(vBfft) / np.max(abs(vX))) 

vBBlockfft = np.fft.fft(vBSequBlock,sNbins)
vBBlockfftMag = 20*sa.safelog10(np.abs(vBBlockfft) / np.max(abs(vX))) 

vBReckFiltfft = np.fft.fft(vWcoeffcut,sNbins)
vBReckFiltfftMag = 20*sa.safelog10(np.abs(vBReckFiltfft) / np.max(abs(vBReckFiltfft))) 

vDiffBlock = vX - vBBlockfft#np.fft.fft(vWcoeffcut,sNbins)#
vDiffBlockMag = 20*sa.safelog10(np.abs(vDiffBlock) / np.max(abs(vX))) 

vBlockRec = np.fft.fft(mRIdeal @ vBSequBlock, sNbins)
vBlockRecMag = 20*sa.safelog10(np.abs(vBlockRec) / np.max(abs(vX))) 

vDiffSingle = vX - vBfft
vDiffSingleMag = 20*sa.safelog10(np.abs(vDiffSingle) / np.max(abs(vDiffSingle))) 

vDiffSDQ = vX - vBSDQfft
vDiffSDQMag = 20*sa.safelog10(np.abs(vDiffSDQ) / np.max(abs(vDiffSDQ))) 

vSingleRec = np.fft.fft(mRIdeal @ vBSequSingle, sNbins)
vSingleRecMag = 20*sa.safelog10(np.abs(vSingleRec) / np.max(abs(vX))) 

# Frequency bins
vFreq = np.fft.fftfreq(sNbins, sT)


# %% [markdown]
# SNR Calculations
# Filtered Signals


vxSigFilt           = np.convolve(vWcoeff,vx,'same')
vxErrFilt           = np.convolve(vWcoeff,(vx-vx),'same')
vbSDQErrFilt        = np.convolve(vWcoeff,(vx-vBSDQ),'same')
vbSequErrFilt       = np.convolve(vWcoeff,(vx-vBSequSingle),'same')
vbBlockSequErrFilt  = np.convolve(vWcoeff,(vx-vBSequBlock),'same')

sVX_MSEIdeal, sVX_SNRdbIdeal, sVX_PSNRdbIdeal = sa.evalN(vxErrFiltIdeal, vxSigFiltIdeal)
sVSDQ_MSEIdeal, sVSDQ_SNRdbIdeal, sVSDQ_PSNRdbIdeal = sa.evalN(vbSDQErrFiltIdeal, vxSigFiltIdeal)
sVB_MSEIdeal, sVB_SNRdbIdeal, sVB_PSNRdbIdeal = sa.evalN(vbSequErrFiltIdeal, vxSigFiltIdeal)
sVBBlock_MSEIdeal, sVBBlock_SNRdbIdeal, sVBBlock_PSNRdbIdeal = sa.evalN(vbBlockSequErrFiltIdeal, vxSigFiltIdeal)    
    
sVX_MSE, sVX_SNRdb, sVX_PSNRdb = sa.evalN(vxErrFilt, vxSigFilt)
sVSDQ_MSE, sVSDQ_SNRdb, sVSDQ_PSNRdb = sa.evalN(vbSDQErrFilt, vxSigFilt)
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

xticks = np.linspace(0, np.pi, 11)
xtick_labels = [r'$0$', r'$\frac{\pi}{10}$', r'$\frac{2\pi}{10}$', r'$\frac{3\pi}{10}$', 
                r'$\frac{4\pi}{10}$', r'$\frac{5\pi}{10}$', r'$\frac{6\pi}{10}$', 
                r'$\frac{7\pi}{10}$', r'$\frac{8\pi}{10}$', r'$\frac{9\pi}{10}$', r'$\pi$']
vNormFrequ = (vFreq / sFs) * 2* np.pi

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
props2 = dict(boxstyle='round', facecolor='lightgray', alpha=0.75)

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
#pltCompSDQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltCompSDQ.plot(vNormFrequ[:sNbins // 2], vBSDQfftMag[:sNbins // 2], color='silver', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2)), linewidth=1.5)
pltCompSDQ.plot(vNormFrequ[:sNbins // 2], vBfftMag[:sNbins // 2], color='black', label='OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)

#pltCompSDQ.set_title(r'Frequency Spectrum', fontsize=13)
pltCompSDQ.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompSDQ.set_xlim([0, np.pi])
pltCompSDQ.set_ylim([-60, 5])
txtStr = 'OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2))
pltCompSDQ.text(1.75, -35, txtStr, fontsize=13, verticalalignment='top', bbox=props)
txtStr = 'SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2))
pltCompSDQ.text(1.75, -47, txtStr, fontsize=13, verticalalignment='top', bbox=props2)
pltCompSDQ.minorticks_on()
pltCompSDQ.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
#pltCompSDQ.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompSDQ.set_xticks(xticks)
pltCompSDQ.set_xticklabels(xtick_labels, fontsize=13)

pltCompOBBQ = figTwo.add_subplot(Pltgs2[1,0])
#pltCompOBBQ.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ', linewidth=1.5)
#pltCompOBBQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltCompOBBQ.plot(vNormFrequ[:sNbins // 2], vBSDQfftMag[:sNbins // 2], color='silver', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2)), linewidth=1.5)
pltCompOBBQ.plot(vNormFrequ[:sNbins // 2], vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompOBBQ.set_title(r'Frequency Spectrum OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), fontsize=14)
pltCompOBBQ.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompOBBQ.set_xlim([0, np.pi])
pltCompOBBQ.set_ylim([-60, 5])
txtStr = 'OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2))
pltCompOBBQ.text(1.75, -35, txtStr, fontsize=13, verticalalignment='top', bbox=props)
txtStr = 'SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2))
pltCompOBBQ.text(1.75, -47, txtStr, fontsize=13, verticalalignment='top', bbox=props2)
pltCompOBBQ.minorticks_on()
pltCompOBBQ.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
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
#pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vDiffSDQMag[:sNbins // 2], color='silver', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)
pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vDiffSingleMag[:sNbins // 2], color='black', label='OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompSDQd.set_title(r'Difference Spectrum', fontsize=13)
pltCompSDQd.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompSDQd.set_xlim([0, np.pi])
pltCompSDQd.set_ylim([-60, 5])
txtStr = 'OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2))
pltCompSDQd.text(1.75, -35, txtStr, fontsize=13, verticalalignment='top', bbox=props)
txtStr = 'SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2))
pltCompSDQd.text(1.75, -47, txtStr, fontsize=13, verticalalignment='top', bbox=props2)
pltCompSDQd.minorticks_on()
pltCompSDQd.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
pltCompSDQd.set_xticks(xticks)
pltCompSDQd.set_xticklabels(xtick_labels, fontsize=13)

pltCompOBBQd = figThree.add_subplot(Pltgs2[1,0])
#pltCompOBBQd.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vDiffBlockMag[:sNbins // 2], color='black', label='OBBQ', linewidth=1.5)
pltCompOBBQd.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=2, linestyle='--')
pltCompOBBQd.plot(vNormFrequ[:sNbins // 2], vDiffSDQMag[:sNbins // 2], color='silver', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)
pltCompOBBQd.plot(vNormFrequ[:sNbins // 2], vDiffBlockMag[:sNbins // 2], color='black', label='OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), linewidth=1.5)
#pltCompOBBQd.set_title(r'Difference Spectrum OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), fontsize=14)
pltCompOBBQd.set_ylabel(r'Magnitude (dB)', fontsize=13)

pltCompOBBQd.set_ylim([-60, 5])
txtStr = 'OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2))
pltCompOBBQd.text(1.75, -35, txtStr, fontsize=13, verticalalignment='top', bbox=props)
txtStr = 'SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2))
pltCompOBBQd.text(1.75, -47, txtStr, fontsize=13, verticalalignment='top', bbox=props2)
pltCompOBBQd.minorticks_on()
pltCompOBBQd.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
pltCompOBBQd.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltCompOBBQd.set_xlim([0, np.pi])
pltCompOBBQd.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompOBBQd.set_xticks(xticks)
pltCompOBBQd.set_xticklabels(xtick_labels, fontsize=13)
# Adjust layout
mtplt.tight_layout(pad=0.25)
mtplt.show()

#####################################

# Create a new figure with a 2x2 layout
figFour = mtplt.figure()  # Double the width for two columns
Pltgs3 = gridspec.GridSpec(2, 2)

# Plot vWcoeff in the first row, first column
pltWcoeff = figFour.add_subplot(Pltgs3[0, 0])
pltWcoeff.plot(vWcoeff, color='#4f97cb', linewidth=1.5)
pltWcoeff.set_title('Filter Coefficients', fontsize=13)
pltWcoeff.set_ylabel('Amplitude', fontsize=13)
pltWcoeff.set_xlabel('Coefficient Index', fontsize=13)
pltWcoeff.legend(fontsize=11)
pltWcoeff.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltWcoeff.minorticks_on()

# Plot vWcoeffcut in the second row, first column
pltWcoeffcut = figFour.add_subplot(Pltgs3[1, 0])
pltWcoeffcut.plot(vWcoeffcut, color='green', linewidth=1.5)
pltWcoeffcut.set_title('Minimum Phase Filter Coefficients', fontsize=13)
pltWcoeffcut.set_ylabel('Amplitude', fontsize=13)
pltWcoeffcut.set_xlabel('Coefficient Index', fontsize=13)
pltWcoeffcut.legend(fontsize=11)
pltWcoeffcut.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltWcoeffcut.minorticks_on()

# Plot Pole-Zero diagram in the first row, second column
pltPZ = figFour.add_subplot(Pltgs3[0, 1])
z, p, k = sigP.tf2zpk(vWcoeff, [1])  # Calculate poles, zeros, and gain
pltPZ.scatter(np.real(z), np.imag(z), s=50, color='#4f97cb', label='Zeros')  # Zeros
pltPZ.scatter(np.real(p), np.imag(p), s=50, color='red', marker='x', label='Poles')  # Poles
pltPZ.axhline(0, color='gray', linewidth=0.7)
pltPZ.axvline(0, color='gray', linewidth=0.7)
pltPZ.set_title('Pole-Zero Diagram', fontsize=13)
pltPZ.set_xlabel('Real Part', fontsize=13)
pltPZ.set_ylabel('Imaginary Part', fontsize=13)
pltPZ.legend(fontsize=11)
pltPZ.grid(True, linestyle='--', linewidth=0.3, color='gray')

# Plot Pole-Zero diagram for vWcoeffcut in the second row, second column
pltPZcut = figFour.add_subplot(Pltgs3[1, 1])
z_cut, p_cut, k_cut = sigP.tf2zpk(vWcoeffcut, [1])  # Calculate poles, zeros, and gain for pruned coefficients
pltPZcut.scatter(np.real(z_cut), np.imag(z_cut), s=50, color='#4f97cb', label='Zeros')  # Zeros
pltPZcut.scatter(np.real(p_cut), np.imag(p_cut), s=50, color='red', marker='x', label='Poles')  # Poles
pltPZcut.axhline(0, color='gray', linewidth=0.7)
pltPZcut.axvline(0, color='gray', linewidth=0.7)
pltPZcut.set_title('Pole-Zero Diagram', fontsize=13)
pltPZcut.set_xlabel('Real Part', fontsize=13)
pltPZcut.set_ylabel('Imaginary Part', fontsize=13)
pltPZcut.legend(fontsize=11)
pltPZcut.grid(True, linestyle='--', linewidth=0.3, color='gray')


figSeven = mtplt.figure()
Pltgs2 = gridspec.GridSpec(3, 1)
# (2,1) Spectrum of the difference signal vDiffSingleMag with overlay of vSingleRecMag
pltCompAdditional = figSeven.add_subplot(Pltgs2[0,0])
#pltCompSDQd.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vDiffSingleMag[:sNbins // 2], color='black', label='SDQ', linewidth=1.5)
#pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltCompAdditional.plot(vNormFrequ[:sNbins // 2], vXMag_org[:sNbins // 2], color='black', linewidth=1.5)
#pltCompSDQd.set_title(r'Difference Spectrum', fontsize=13)
pltCompAdditional.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompAdditional.set_xlim([0, np.pi])
pltCompAdditional.set_ylim([-90, 5])
pltCompAdditional.set_title('Original signal', fontsize=13)
pltCompAdditional.minorticks_on()
pltCompAdditional.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltCompAdditional.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltCompAdditional.set_xlim([0, np.pi])
pltCompAdditional.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompAdditional.set_xticks(xticks)
pltCompAdditional.set_xticklabels(xtick_labels, fontsize=13)

pltCompAdditionalO = figSeven.add_subplot(Pltgs2[1,0])
#pltCompSDQd.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vDiffSingleMag[:sNbins // 2], color='black', label='SDQ', linewidth=1.5)
#pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltCompAdditionalO.plot(vNormFrequ[:sNbins // 2], vXMag[:sNbins // 2], color='black', linewidth=1.5)
#pltCompSDQd.set_title(r'Difference Spectrum', fontsize=13)
pltCompAdditionalO.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompAdditionalO.set_xlim([0, np.pi])
pltCompAdditionalO.set_ylim([-90, 5])
pltCompAdditionalO.set_title('Filtered input signal', fontsize=13)
pltCompAdditionalO.minorticks_on()
pltCompAdditionalO.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltCompAdditionalO.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltCompAdditionalO.set_xlim([0, np.pi])
pltCompAdditionalO.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompAdditionalO.set_xticks(xticks)
pltCompAdditionalO.set_xticklabels(xtick_labels, fontsize=13)

pltCompAdditional1 = figSeven.add_subplot(Pltgs2[2,0])
#pltCompSDQd.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vDiffSingleMag[:sNbins // 2], color='black', label='SDQ', linewidth=1.5)
#pltCompSDQd.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltCompAdditional1.plot(vNormFrequ[:sNbins // 2], vBlockRecMag[:sNbins // 2], color='black', linewidth=1.5)
#pltCompSDQd.set_title(r'Difference Spectrum', fontsize=13)
pltCompAdditional1.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltCompAdditional1.set_xlim([0, np.pi])
pltCompAdditional1.set_ylim([-90, 5])
pltCompAdditional1.set_title('Reconstructed signal', fontsize=13)
pltCompAdditional1.minorticks_on()
pltCompAdditional1.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
pltCompAdditional1.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltCompAdditional1.set_xlim([0, np.pi])
pltCompAdditional1.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltCompAdditional1.set_xticks(xticks)
pltCompAdditional1.set_xticklabels(xtick_labels, fontsize=13)

# Adjust layout
mtplt.tight_layout(pad=0.25)
mtplt.show()

###### POSTER ISCAS25######
# Convert to inches
width_cm = 12.5  # Desired width in cm
height_cm = 5  # Desired height in cm
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
figTwo = mtplt.figure(figsize=(width_inch,height_cm))
Pltgs2 = gridspec.GridSpec(3, 1)
# (2,1) Spectrum of the difference signal vDiffSingleMag with overlay of vSingleRecMag
pltSDQ = figTwo.add_subplot(Pltgs2[0,0])
#pltCompSDQ.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vBfftMag[:sNbins // 2], color='black', label='SDQ', linewidth=1.5)
#pltCompSDQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
pltSDQ.plot(vNormFrequ[:sNbins // 2], vBSDQfftMag[:sNbins // 2], color='gray', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2)), linewidth=1.5)
#pltSDQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=2, linestyle='--')
#pltCompSDQ.plot(vNormFrequ[:sNbins // 2], vBfftMag[:sNbins // 2], color='black', label='OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVB_SNRdbIdeal, 2)), linewidth=1.5)

#pltCompSDQ.set_title(r'Frequency Spectrum', fontsize=13)
pltSDQ.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltSDQ.set_xlim([0, np.pi])
pltSDQ.set_ylim([-60, 5])
#txtStr = 'OBAQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2))
txtStr = '(LP) ' + '$\Sigma\Delta$' + '-SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2))
pltSDQ.text(1.2, -44, txtStr, fontsize=13, verticalalignment='top', bbox=props2)
pltSDQ.minorticks_on()
pltSDQ.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
#pltCompSDQ.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltSDQ.set_xticks(xticks)
pltSDQ.set_xticklabels(xtick_labels, fontsize=13)

pltOBBQ = figTwo.add_subplot(Pltgs2[1,0])
#pltCompOBBQ.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ', linewidth=1.5)
#pltCompOBBQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
#pltOBBQ.plot(vNormFrequ[:sNbins // 2], vBSDQfftMag[:sNbins // 2], color='silver', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2)), linewidth=1.5)
pltOBBQ.plot(vNormFrequ[:sNbins // 2], vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), linewidth=1.5)
pltOBBQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=2, linestyle='--')
#pltCompOBBQ.set_title(r'Frequency Spectrum OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), fontsize=14)
pltOBBQ.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltOBBQ.set_xlim([0, np.pi])
pltOBBQ.set_ylim([-60, 5])
txtStr = '(LP) OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2))
pltOBBQ.text(1.2, -46, txtStr, fontsize=13, verticalalignment='top', bbox=props)
pltOBBQ.minorticks_on()
pltOBBQ.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
pltOBBQ.set_xlim([0,np.pi])
pltOBBQ.set_xticks(xticks)
pltOBBQ.set_xticklabels(xtick_labels, fontsize=13)

vBBlockfftMag_BP = np.load('saves/vBBlockfftMag.npy').flatten()
vBReckFiltfftMag_BP = np.load('saves/vBReckFiltfftMag.npy').flatten()
sVBBlock_SNRdbIdeal_BP = np.load('saves/sVBBlock_SNRdbIdeal.npy').flatten()

pltOBBQBP = figTwo.add_subplot(Pltgs2[2,0])
#pltCompOBBQ.plot(vFreq[:sNbins // 2] / (sFs / sNbins) * np.pi, vBBlockfftMag[:sNbins // 2], color='black', label='OBBQ', linewidth=1.5)
#pltCompOBBQ.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag[:sNbins // 2], color='#4f97cb', linewidth=1.5)
#pltOBBQ.plot(vNormFrequ[:sNbins // 2], vBSDQfftMag[:sNbins // 2], color='silver', label='SDQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVSDQ_SNRdbIdeal, 2)), linewidth=1.5)
pltOBBQBP.plot(vNormFrequ[:sNbins // 2], vBBlockfftMag_BP[:sNbins // 2], color='black', label='OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), linewidth=1.5)
pltOBBQBP.plot(vNormFrequ[:sNbins // 2], vBReckFiltfftMag_BP[:sNbins // 2], color='#4f97cb', linewidth=2, linestyle='--')
#pltCompOBBQ.set_title(r'Frequency Spectrum OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal, 2)), fontsize=14)
pltOBBQBP.set_ylabel(r'Magnitude (dB)', fontsize=13)
pltOBBQBP.set_xlim([0, np.pi])
pltOBBQBP.set_ylim([-60, 5])
txtStr = '(BP) OBBQ, SER: {snr_ideal} dB'.format(snr_ideal=round(sVBBlock_SNRdbIdeal_BP[0], 2))
pltOBBQBP.text(1.2, -46, txtStr, fontsize=13, verticalalignment='top', bbox=props)
pltOBBQBP.minorticks_on()
pltOBBQBP.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgray')
pltOBBQBP.set_xlabel(r'Frequency (Hz)', fontsize = 13)
pltOBBQBP.set_xlim([0,np.pi])
pltOBBQBP.set_xlabel(r'Normalized Frequency (radians/sample)', fontsize=13)
pltOBBQBP.set_xticks(xticks)
pltOBBQBP.set_xticklabels(xtick_labels, fontsize=13)
# Adjust layout
mtplt.tight_layout(pad=0.25)
mtplt.show()

