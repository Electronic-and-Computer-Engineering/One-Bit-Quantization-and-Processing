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

sN = 4096    
sBSize = 32

#sHop = sBSize

# %% [markdown]
# Generate input signal
sNewSignal = True
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
v_n = np.arange(sN).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# %%
### Generate ideal matrices ###
vLPRangeFs          = [0, 0, sSigFmax, sSigFmax]
vLPfilterRad        = sg.freq2rad(vLPRangeFs, sFs)
vRIdeal, vRIdealShift, vW = filt.idealBinFilt(sN, vLPfilterRad, 1.0, 0.0, True)
#vRIdeal, vW = filt.idealBinFilt(sN, sg.freq2bin(sSigFmax, sN, sFs), sMinBin=None, sType='lowpass', full=True)
mRIdeal = scLinAlg.toeplitz(vRIdeal)

sFpb        = sSigFmax
sFsb        = sFpb + 16
sApbdB      = 0.001
sAsbdB      = 40

# create Filter
(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False)
np.save('vWcoeff.npy', vWcoeff)

sP = 1024
vbPWM  = obq.pwmQuant(vx, sP, "triangle")
vbOPWM = obq.OptPWM(vx, vWcoeff, sP, 0, fix_Mp=True, fix_kappa=False)


## %%
## Analysis ##
with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
    vX              = np.fft.fft(mRIdeal @ vx)
    vxSigFiltIdeal  = mRIdeal @ vx
    vxErrFiltIdeal  = mRIdeal @ (vx - vx)
    vbOPWMErrFiltIdeal = mRIdeal @ (vx - vbOPWM)
    vbPWMErrFiltIdeal = mRIdeal @ (vx - vbPWM)


vBOPWMfft       = np.fft.fft(vbOPWM,sN)
vBPWMfft        = np.fft.fft(vbPWM,sN)
vXMag           = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))
vBPWMMag        = 20*sa.safelog10(np.abs(vBPWMfft) / np.max(abs(vX)))
vOBPWMMag        = 20*sa.safelog10(np.abs(vBOPWMfft) / np.max(abs(vX)))

# Frequency bins
vFreq = np.fft.fftfreq(sN, sT)


vxSigFilt               = np.convolve(vWcoeff,vx,'same')
vxErrFilt               = np.convolve(vWcoeff,(vx-vx),'same')
vbOPWMErrFilt            = np.convolve(vWcoeff,(vx-vbOPWM),'same')
vbPWMErrFilt            = np.convolve(vWcoeff,(vx-vbPWM),'same')

sVX_MSE, sVX_SNRdb, sVX_PSNRdb = sa.evalN(vxErrFilt, vxSigFilt)
sVX_MSEIdeal, sVX_SNRdbIdeal, sVX_PSNRdbIdeal = sa.evalN(vxErrFiltIdeal, vxSigFiltIdeal)

sVBPWM_MSE, sVBPWM_SNRdb, sVBPWM_PSNRdb = sa.evalN(vbPWMErrFilt, vxSigFilt)
sVBPWM_MSEIdeal, sVBPWM_SNRdbIdeal, sVBPWM_PSNRdbIdeal = sa.evalN(vbPWMErrFiltIdeal, vxSigFiltIdeal)

sVBOPWM_MSE, sVBOPWM_SNRdb, sVBOPWM_PSNRdb = sa.evalN(vbOPWMErrFilt, vxSigFilt)
sVBOPWM_MSEIdeal, sVBOPWM_SNRdbIdeal, sVBOPWM_PSNRdbIdeal = sa.evalN(vbOPWMErrFiltIdeal, vxSigFiltIdeal)

## Plotting ###
vNormFrequ = np.linspace(0, 2*np.pi, sN)
xticks = np.linspace(0, 2*np.pi, 21)
xtick_labels = [r'$0$', r'$\frac{\pi}{10}$', r'$\frac{2\pi}{10}$', r'$\frac{3\pi}{10}$', 
                r'$\frac{4\pi}{10}$', r'$\frac{5\pi}{10}$', r'$\frac{6\pi}{10}$', 
                r'$\frac{7\pi}{10}$', r'$\frac{8\pi}{10}$', r'$\frac{9\pi}{10}$', r'$\pi$', 
                r'$\frac{11\pi}{10}$', r'$\frac{12\pi}{10}$', r'$\frac{13\pi}{10}$', 
                r'$\frac{14\pi}{10}$', r'$\frac{15\pi}{10}$', r'$\frac{16\pi}{10}$', 
                r'$\frac{17\pi}{10}$', r'$\frac{18\pi}{10}$', r'$\frac{19\pi}{10}$', r'$2\pi$']

figOne = mtplt.figure()
Pltgs = gridspec.GridSpec(2,1)

# =============================================================================
# pltDiscTime = figOne.add_subplot(Pltgs[0,:])
# pltDiscTime.plot(vx)
# pltDiscTime.set_title('Input Signal')
# pltDiscTime.set_xlabel('Samples $n$', fontsize = 11)
# pltDiscTime.set_ylabel('Amplitude', fontsize = 11)
# pltDiscTime.set_xlim([0,sN])
# mtplt.minorticks_on()
# mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
# 
# pltFreq = figOne.add_subplot(Pltgs[1,:])
# pltFreq.plot(vNormFrequ, vXMag)
# #pltFreq.set_title('Input Signal')
# #pltFreq.set_xlabel('normalized Frequency', fontsize = 11)
# pltFreq.set_title('Frequency Spectrum vX SER: {snr} dB\nFrequency Spectrum vX Ideal: {snr_ideal} dB'.format(snr=round(sVX_SNRdb, 2), snr_ideal=round(sVX_SNRdbIdeal, 2)))
# pltFreq.set_ylabel('Magnitude (dB)', fontsize = 11)
# pltFreq.set_xlim([0,2*np.pi])
# pltFreq.set_ylim([-60,10])
# pltFreq.set_xticks(xticks)
# pltFreq.set_xticklabels(xtick_labels, fontsize=13)
# mtplt.minorticks_on()
# mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')
# =============================================================================

pltFreq = figOne.add_subplot(Pltgs[0,:])
pltFreq.plot(vNormFrequ, vBPWMMag)
#pltFreq.set_title('Input Signal')
#pltFreq.set_xlabel('normalized Frequency', fontsize = 11)
pltFreq.set_title('Frequency Spectrum PWM SER: {snr} dB\nFrequency Spectrum oPWM Ideal: {snr_ideal} dB'.format(snr=round(sVBPWM_SNRdb, 2), snr_ideal=round(sVBPWM_SNRdbIdeal, 2)))
pltFreq.set_ylabel('Magnitude (dB)', fontsize = 11)
pltFreq.set_ylim([-60,10])
pltFreq.set_xlim([0,2*np.pi])
pltFreq.set_xticks(xticks)
pltFreq.set_xticklabels(xtick_labels, fontsize=13)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreq = figOne.add_subplot(Pltgs[1,:])
pltFreq.plot(vNormFrequ, vOBPWMMag)
#pltFreq.set_title('Input Signal')
#pltFreq.set_xlabel('normalized Frequency', fontsize = 11)
pltFreq.set_title('Frequency Spectrum oPWM SER: {snr} dB\nFrequency Spectrum oPWM Ideal: {snr_ideal} dB'.format(snr=round(sVBOPWM_SNRdb, 2), snr_ideal=round(sVBOPWM_SNRdbIdeal, 2)))
pltFreq.set_ylabel('Magnitude (dB)', fontsize = 11)
pltFreq.set_ylim([-60,10])
pltFreq.set_xlim([0,2*np.pi])
pltFreq.set_xticks(xticks)
pltFreq.set_xticklabels(xtick_labels, fontsize=13)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

figOne.set_tight_layout(True)
