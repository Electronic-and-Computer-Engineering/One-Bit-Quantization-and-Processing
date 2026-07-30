# %%
# system packages
import numpy as np

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
import sys, os, time
sys.path.append('../01_Library')
# individual packages
import sg, sa, sp, obq, filt 
import globalTools

mtplt.close('all')
np.seterr(all='ignore')

def _energy_centroid(xm: np.ndarray) -> float:
    # c_eng = sum(n*x[n]^2)/sum(x[n]^2)
    vE = xm * xm
    tot = vE.sum()
    if tot <= 0:
        return (len(xm) - 1) / 2.0
    idx = np.arange(len(xm), dtype=float)
    return float((idx @ vE) / tot)

def _cut_from_centroid(c_eng: float, M: int) -> int:
    # sIdxCut = round(M/2 - c_eng), clipped to [0, M]
    s = int(round(M/2 - c_eng))
    return int(np.clip(s, 0, M))

def analyze_example_to_npy(vx: np.ndarray, M: int, H: int, save_path: str):
    """
    Split vx into blocks (length M, hop H), compute per-block:
      - centroid (energy center)
      - cut_idx (derived from centroid)
    Save a dict to .npy with: vx, M, H, blocks, centroid, cut_idx.
    """
    N = len(vx)
    if M <= 0 or H <= 0 or M > N:
        raise ValueError("Invalid M/H relative to signal length.")
    num_blocks = (N - M) // H + 1
    starts = np.arange(num_blocks) * H

    blocks   = np.empty((num_blocks, M), dtype=float)
    centroid = np.empty(num_blocks, dtype=float)
    cut_idx  = np.empty(num_blocks, dtype=int)

    for i, st in enumerate(starts):
        xm = vx[st:st+M]
        c  = _energy_centroid(xm)
        k  = _cut_from_centroid(c, M)
        blocks[i, :] = xm
        centroid[i]  = c
        cut_idx[i]   = k

    out = {
        "vx": vx,
        "M": M,
        "H": H,
        "blocks": blocks,
        "centroid": centroid,
        "cut_idx": cut_idx,
        "starts": starts,
    }
    np.save(save_path, out, allow_pickle=True)
    return out  # auch direkt im RAM verfügbar

sNbins    = 2048

#sBlockSize = 16 #32
sM         = 64 #16
sL_max     = sNbins
sHop       = 16
sK         = 111

sBSize = 16

# Generate input signal
###############################################
sNewSignal = True ###################  #######
### Signal generation ###
if sNewSignal:
    sFs = 2048
    sSigFmax = 111
    vxFrequ = (np.arange(1, sSigFmax, step=1)).reshape(-1, 1)
    vxPhase = np.random.rand(len(vxFrequ), 1) * 2 * np.pi

    np.save('saves/vxFrequ.npy', vxFrequ)
    np.save('saves/vxPhase.npy', vxPhase)
    np.save('saves/sSigFmax.npy', sSigFmax)
    np.save('saves/sFs.npy', sFs)
else:    
    vxFrequ     = np.load('saves/vxFrequ.npy')
    vxPhase     = np.load('saves/vxPhase.npy')
    sSigFmax    = np.load('saves/sSigFmax.npy')
    sFs         = np.load('saves/sFs.npy') 

sT = 1 / (sFs)
v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

out = analyze_example_to_npy(vx, M=32, H=16, save_path="example1.npy")

#sQBit = 16
#sLsB = (2)/(2**sQBit)
#vx = (np.round(vx/sLsB))/(2**(sQBit-1))
################################################
##########  ########  #######  #################
