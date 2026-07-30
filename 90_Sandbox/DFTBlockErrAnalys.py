import numpy as np
import scipy.linalg as scLinAlg
import scipy.signal as sigP
from scipy.fftpack import hilbert

import os
import sg, sa, sp, obq, filt

# === CONFIGURATION ===
sSaveFolder = 'results_vx_dftq'
os.makedirs(sSaveFolder, exist_ok=True)

sNbins = 256
sFs = 256
sSigFmax = 42

sK_list = [1, 4, 16, 32, 64, 128]
sM_list = [1, 4, 8, 16, 32]
sHop_list = [1, 2, 4, 8]
sL_max = sNbins

# === SIGNAL GENERATION ===
sNewSignal = True
if sNewSignal:
    vxFrequ = (np.arange(0, sSigFmax, step=1)).reshape(-1, 1)
    vxPhase = np.random.rand(len(vxFrequ), 1) * 2 * np.pi
    np.save('saves/vxFrequ.npy', vxFrequ)
    np.save('saves/vxPhase.npy', vxPhase)
else:
    vxFrequ = np.load('saves/vxFrequ.npy')
    vxPhase = np.load('saves/vxPhase.npy')

v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# === FILTERS ===
vLPRangeFs = [0, 0, sSigFmax, sSigFmax]
vLPfilterRad = sg.freq2rad(vLPRangeFs, sFs)
vLPfilterRadIdeal = sg.freq2rad(vLPRangeFs, sFs)

vRIdeal, vRIdealShift, vW = filt.idealBinFilt(sNbins, vLPfilterRadIdeal, 1.0, 0.0, True)
mRIdeal = scLinAlg.toeplitz(vRIdeal)

# === MAIN LOOP ===
for sK in sK_list:
    vW_D = sp.getFiltWeights(vW, sK, vLPfilterRad)

    for sM in sM_list:
        for sHop in sHop_list:
            print(f"Running sK={sK}, sM={sM}, sHop={sHop}")
            try:
                vBDFT, vBlockErr, mErr_k_p = obq.iterBlockQDFT(vx, vLPfilterRad, vW_D, sK, sM, sL_max, sHop, 'grb', False, 1)

                fname_prefix = f"DFTQ_sK{sK}_sM{sM}_sHop{sHop}"
                np.save(os.path.join(sSaveFolder, fname_prefix + '_vBDFT.npy'), vBDFT)
                np.save(os.path.join(sSaveFolder, fname_prefix + '_vBlockErr.npy'), vBlockErr)
                np.save(os.path.join(sSaveFolder, fname_prefix + '_mErr_k_p.npy'), mErr_k_p)
            except Exception as e:
                print(f"Failed for sK={sK}, sM={sM}, sHop={sHop}: {e}")

print("All configurations completed.")