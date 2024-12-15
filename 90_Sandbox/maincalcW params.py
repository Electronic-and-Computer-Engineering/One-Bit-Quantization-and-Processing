import numpy as np
import scipy.signal as sigP
import scipy.linalg as scLinAlg

from scipy.optimize import differential_evolution

import matplotlib.pyplot as mtplt
import sg, sa, sp, obq, misc, filt
import os, sys

# Initial setup parameters
sNbins = 4096
sFs = 4096
sBSize = 8
sSigFmax = 63
bSNRideal = True

# Signal generation
v_n = np.arange(sNbins).reshape(-1, 1)
vxFrequ = (np.arange(0, sSigFmax, step=2)).reshape(-1, 1)
vxPhaseInit = np.random.rand(len(vxFrequ), 1) * 2 * np.pi
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhaseInit, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# Initial quantization
vBSequSingle, ve, ve_hat = obq.iterSequQ(vx, np.tril(np.ones((sNbins, sNbins))), 0)

# Initialize the best improvement value and iteration counter
best_ser = -np.inf
best_params = None
iteration = 0

### Generate ideal matrices ###
vRIdeal = sp.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), 'normal')
mRIdeal = scLinAlg.toeplitz(vRIdeal)


# Define the objective function to maximize sVBBlock_SERdb
def calculate_ser(params):
    global best_ser, best_params, iteration
    iteration += 1
    
    sFpb, sFsb, sApb, sAsb = params
    # create Filter
    sTaps = filt.firFindOptN(sFs, sFpb, sFsb, sApb, sAsb)
    (vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calc(sFs, sFpb, sFsb, sApb, sAsb, sTaps)
    ##plotFilter 
    filt.plotFrequResp(vw, vH, sFs, sFpb, sFsb, sHpbMin, sHpbMax, sHsbMax)
    mtplt.tight_layout()
    mtplt.pause(1)
    vW = vWcoeff / np.sum(vWcoeff)
    
    vBSequBlock, vEBlock = obq.iterBlockQnew(vx, vW, sBSize, sBSize, 'grb')
    
    if bSNRideal: 
        vxSigFilt           = mRIdeal @ vx
        vxErrFilt           = mRIdeal @ (vx - vx)
        vbSequErrFilt       = mRIdeal @ (vx-vBSequSingle)
        vbBlockSequErrFilt  = mRIdeal @ (vx-vBSequBlock)
    else:
        vxSigFilt           = np.convolve(vW, vx, 'same')
        vxErrFilt           = np.convolve(vW, (vx - vx), 'same')
        vbSequErrFilt       = np.convolve(vW, (vx - vBSequSingle), 'same')
        vbBlockSequErrFilt  = np.convolve(vW, (vx - vBSequBlock), 'same')
    
    _, sVB_SERdb, _ = sa.evalN(vbSequErrFilt, vxSigFilt)
    _, sVBBlock_SERdb, _ = sa.evalN(vbBlockSequErrFilt, vxSigFilt)
    
    current_ser = sVBBlock_SERdb
    
    if current_ser > best_ser:
        best_ser = current_ser
        best_params = params
    
    print(f"Run {iteration}")
    print(f"Current Parameters: sFpb={sFpb:.2f}, sFsb={sFsb:.2f}, sApass={sApb:.2f}, sAstop={sAsb:.2f}")
    print(f"Current SER: {current_ser:.2f} dB (Block) | {sVB_SERdb:.2f} dB (Single)")
    print(f"Best SER so far: {best_ser:.2f} dB for Parameters: sFpb={best_params[0]:.2f}, sFsb={best_params[1]:.2f}, sApass={best_params[2]:.2f}, sAstop={best_params[3]:.2f}")
    
    return -current_ser  # Minimize the negative SER to maximize the SER

# Define bounds for the parameters: (sPassHz, sTransWHz, sApass, sAstop)
bounds = [
    (sSigFmax - 2, sSigFmax + 75),  # sPassHz
    (10, 100),       # sTransWHz
    (0, 1),          # sApass
    (20, 100)        # sAstop
]

# Run the differential evolution optimization
result = differential_evolution(
    calculate_ser,
    bounds,
    strategy='best2bin',
    maxiter=1000,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    disp=True
)

# Extract the optimal values
optimal_sFpb, optimal_sFsb, optimal_sApass, optimal_sAstop = result.x
optimal_ser = -result.fun  # Convert back to positive

print(f"Optimal Passband Frequency:     {optimal_sFpb} Hz")
print(f"Optimal Stopband Frequency:     {optimal_sFsb} Hz")
print(f"Optimal Passband Ripple:        {optimal_sApass} dB")
print(f"Optimal Stopband Attenuation:   {optimal_sAstop} dB")
print(f"Optimal SER: {optimal_ser:.2f} dB")
