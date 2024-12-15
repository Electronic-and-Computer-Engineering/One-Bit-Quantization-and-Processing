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

#sympi
import sympy as sypy

# %%
# individual packages
import sg, sa, sp, obq, sym, misc
strFileNameTxt = "iterBlockCalc.txt"
mtplt.close('all')

sNbins = 2**4
sFs = sNbins
sT = 1 / (sFs)
       
sL = 15
sBSize = 4
sPadLen = sNbins + (sL - 1)
sRecFilterFrequ = 4 

# %% [markdown]
# Generate input signal

# %%
### Signal generation ###
v_n = np.arange(sNbins).reshape(-1, 1)
vxFrequ = (np.arange(0, 4, step=1) * sFs / sNbins).reshape(-1, 1)
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
vRIdeal = sp.idealBinFilt(sNbins, sg.freq2Bin(sRecFilterFrequ, sNbins, sFs), 'normal')
vRIdealvTest = sp.idealBinFilt(42, sg.freq2Bin(sRecFilterFrequ, 42, sFs), 'normal')
mRIdeal = scLinAlg.toeplitz(vRIdeal)

sAtten = 60 # in dB
sBeta = sigP.kaiser_beta(sAtten)
vW = sigP.firwin(sL, (sRecFilterFrequ) / (sFs/2), window=('kaiser', sBeta))

vLsBand = np.array([0, sRecFilterFrequ, sRecFilterFrequ+2, (sFs/2)]) / (sFs/2)
vDesiredGains = np.array([1, 1, 10**(-6), 10**(-6)])
vWLs = sigP.firls(sL, vLsBand, vDesiredGains)
vNormWLs = vWLs / np.sum(vWLs)

# Replace zeros with eps
#vW[vW == 0] = np.finfo(float).eps

mW = sp.convmtx(vW,len(vx),'colWise')
mWNN = sp.convmtx(vW,len(vx),'colWiseNN')

sPadAdd = int(sPadLen - sNbins)
vxPadded = np.pad(vx,(0,sPadAdd))

vymat = mW @ vx
sCut = int(np.floor((sL-1)/2))


mOnes = np.ones((sNbins,sNbins))
mSigDeltaFilt = np.tril(mOnes)

# Symbolic =============================
symVx = sypy.symbols(f'x_0:{sNbins}')
symVx = sypy.Matrix(symVx)
symVw = sypy.symbols(f'w_0:{sL}')
symVw = sypy.Matrix(symVw)
symMW = sym.convmtx(symVw,len(vx),'colWise')

ltxVw = sypy.latex(symVw, mat_delim='')
ltxVx = sypy.latex(symVx, mat_delim='')
ltxMW = sypy.latex(symMW, mat_delim='', mat_str='bmatrix')

txt = f"\\underline{{w}} = {ltxVw}\n\n"
misc.printToTxt(txt, strFileNameTxt)
txt = f"\\underline{{x}} = {ltxVx}\n\n"
misc.printToTxt(txt, strFileNameTxt, True)
txt = f"W = {ltxMW}\n\n"
misc.printToTxt(txt, strFileNameTxt, True)

sym.iterBlockQ(symVx, symVw, vx, vWLs, sBSize, strFileNameTxt)

