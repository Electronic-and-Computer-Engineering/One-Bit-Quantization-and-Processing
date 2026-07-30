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

sNbins = 512    
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
sFsb        = sFpb + 8
sApbdB      = 0.001
sAsbdB      = 40

# create Filter
(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False)
np.save('vWcoeff.npy', vWcoeff)

vbPWM = obq.OptPWM(vx, vWcoeff, 32, 0)

