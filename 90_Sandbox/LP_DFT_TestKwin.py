# %%
# system packages
import numpy as np

#Plotting
import matplotlib.pyplot as mtplt
import pandas as pd

#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
from datetime import datetime

ts = datetime.now().strftime('%Y%m%d_%H%M%S')


mtplt.rcParams['mathtext.fontset'] = 'stix'
mtplt.rcParams['font.family'] = 'STIXGeneral'

# %%
import sys, os
sys.path.append('../01_Library')
# individual packages
import sg, sa, sp, obq, filt 

## =================== ANALYSIS 
# ------------------------------------------------------------------
# Diese Funktion am Ende jedes winIdx-Loops aufrufen:
#
#   log_sweep_result(winIdx, vBDFT, vBlockErr, vBSequBlock,
#                    vx, mRIdeal, vWcoeff, sM, sHop, sNbins,
#                    sK, sSigFmax, sSigFmin, sL_max)
# ------------------------------------------------------------------

def log_sweep_result(winIdx, vBlockErr,
                     sVB_SNRdbIdeal, sVBBlock_SNRdbIdeal,
                     sVB_SNRdb,      sVBBlock_SNRdb,
                     sM, sHop, sNbins, sK, sWLen,
                     sSigFmin, sSigFmax, sL_max):
    """
    Logt ein winIdx-Ergebnis. Alle SER-Werte kommen direkt
    aus deiner bestehenden Evaluation.

    Aufruf am Ende des winIdx-Loops:
        log_sweep_result(winIdx, vBlockErr,
                         sVB_SNRdbIdeal, sVBBlock_SNRdbIdeal,
                         sVB_SNRdb,      sVBBlock_SNRdb,
                         sM, sHop, sNbins, sK, 4*sM,
                         sSigFmin, sSigFmax, sL_max)
    """
    row = {
        # Settings
        'winIdx':              winIdx,
        'k_theory':            sM // 2 - sHop,
        'M':                   sM,
        'Hop':                 sHop,
        'N':                   sNbins,
        'K':                   sK,
        'wLen':                sWLen,
        'Fmin_Hz':             sSigFmin,
        'Fmax_Hz':             sSigFmax,
        'sL':                  sL_max,
        # SER (ideal filter)
        'SER_DFTQ_ideal':      round(sVB_SNRdbIdeal,      3),
        'SER_OBBQ_ideal':      round(sVBBlock_SNRdbIdeal,  3),
        'delta_ideal':         round(sVB_SNRdbIdeal - sVBBlock_SNRdbIdeal, 3),
        # SER (FIR filter)
        'SER_DFTQ_fir':        round(sVB_SNRdb,            3),
        'SER_OBBQ_fir':        round(sVBBlock_SNRdb,       3),
        'delta_fir':           round(sVB_SNRdb - sVBBlock_SNRdb, 3),
        # Block error statistics
        'blockErr_mean':       round(float(np.mean(vBlockErr)),  4),
        'blockErr_std':        round(float(np.std(vBlockErr)),   4),
        'blockErr_max':        round(float(np.max(vBlockErr)),   4),
        'blockErr_trend':      round(float(np.polyfit(
                                   np.arange(len(vBlockErr)),
                                   vBlockErr, 1)[0]), 6),
    }

    df_new = pd.DataFrame([row])
    if os.path.exists(CSV_PATH):
        df_new.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df_new.to_csv(CSV_PATH, mode='w', header=True, index=False)

    print(f"\n winIdx={winIdx:+3d} | "
          f"SER_DFTQ={sVB_SNRdbIdeal:.2f} dB | "
          f"OBBQ={sVBBlock_SNRdbIdeal:.2f} dB | "
          f"delta={sVB_SNRdbIdeal - sVBBlock_SNRdbIdeal:+.2f} dB | "
          f"blockErr mean={np.mean(vBlockErr):.1f} "
          f"trend={row['blockErr_trend']:+.5f}")


def plot_sweep():
    """Plottet alle geloggten Ergebnisse."""
    if not os.path.exists(CSV_PATH):
        print("Keine Ergebnisse gefunden.")
        return

    df  = pd.read_csv(CSV_PATH).sort_values('winIdx')
    k_th = int(df['k_theory'].iloc[0])

    fig, axes = mtplt.subplots(3, 1, figsize=(12, 10))

    # -- SER vs winIdx --
    ax = axes[0]
    ax.plot(df['winIdx'], df['SER_DFTQ_ideal'], 'o-', lw=1.5,
            label='DFTQ (ideal filter)')
    ax.plot(df['winIdx'], df['SER_DFTQ_fir'],   's--', lw=1.2,
            label='DFTQ (FIR filter)')
    ax.axhline(df['SER_OBBQ_ideal'].iloc[0], color='red', ls='--', lw=1.5,
               label=f"OBBQ ideal ({df['SER_OBBQ_ideal'].iloc[0]:.1f} dB)")
    ax.axvline(k_th, color='gray', ls=':', lw=1.2,
               label=f'k_theory = M/2 - Hop = {k_th}')
    best = df.loc[df['SER_DFTQ_ideal'].idxmax()]
    ax.annotate(f"best winIdx={int(best.winIdx)}\n{best.SER_DFTQ_ideal:.2f} dB",
                xy=(best.winIdx, best.SER_DFTQ_ideal),
                xytext=(best.winIdx + 1, best.SER_DFTQ_ideal - 2),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    ax.set_xlabel('winIdx (window shift k)')
    ax.set_ylabel('SER (dB)')
    ax.set_title(f"SER vs. winIdx  |  M={df['M'].iloc[0]}, "
                 f"Hop={df['Hop'].iloc[0]}, wLen={df['wLen'].iloc[0]}, "
                 f"K={df['K'].iloc[0]}, Fmax={df['Fmax_Hz'].iloc[0]} Hz")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- blockErr mean ± std --
    ax = axes[1]
    ax.plot(df['winIdx'], df['blockErr_mean'], 'o-', lw=1.5, label='mean blockErr')
    ax.fill_between(df['winIdx'],
                    df['blockErr_mean'] - df['blockErr_std'],
                    df['blockErr_mean'] + df['blockErr_std'],
                    alpha=0.2, label='±1 std')
    ax.set_xlabel('winIdx')
    ax.set_ylabel('Block Error')
    ax.set_title('Block Error Statistics vs. winIdx')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- blockErr trend (drift) --
    ax = axes[2]
    ax.plot(df['winIdx'], df['blockErr_trend'], 'o-', lw=1.5, color='orange')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel('winIdx')
    ax.set_ylabel('Slope (drift per block)')
    ax.set_title('Block Error Drift  (0 = stable, + = accumulating)')
    ax.grid(True, alpha=0.3)

    mtplt.tight_layout()
    mtplt.savefig('analysis/k_sweep/sweep_plot.png', dpi=120)
    mtplt.show()
    print(f"\nBest winIdx={int(best.winIdx)}  SER={best.SER_DFTQ_ideal:.2f} dB")
    print(f"k_theory = {k_th}")
    print(f"Saved: analysis/k_sweep/sweep_plot.png")


def clear_results():
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
        print("CSV gelöscht.")

### =====================================================

mtplt.close('all')
np.seterr(all='ignore')


os.makedirs('analysis/k_sweep', exist_ok=True)
CSV_PATH = f'analysis/k_sweep/results_{ts}.csv'

sNbins    = 2048
sM         = 32
sL_max     = sNbins
sHop       = 8
sK         = 512

sBSize     = 16

 
# Generate input signal
###############################################
sNewSignal = True ###################  #######
### Signal generation ###
if sNewSignal:
    sFs = 2048
    sSigFmax = 43
    sSigFmin = 1
    sSigFNum = (sSigFmax-sSigFmin)+1
    vxFrequ = (np.linspace(sSigFmin, sSigFmax, num=sSigFNum)).reshape(-1, 1)
    vxPhase = np.random.rand(len(vxFrequ), 1) * 2 * np.pi

    np.save('saves/vxFrequ.npy', vxFrequ)
    np.save('saves/vxPhase.npy', vxPhase)
    np.save('saves/sSigFmax.npy', sSigFmax)
    np.save('saves/sSigFmin.npy', sSigFmin)
    np.save('saves/sFs.npy', sFs)
else:    
    vxFrequ     = np.load('saves/vxFrequ.npy')
    vxPhase     = np.load('saves/vxPhase.npy')
    sSigFmax    = np.load('saves/sSigFmax.npy')
    sSigFmin    = np.load('saves/sSigFmin.npy')
    sFs         = np.load('saves/sFs.npy') 

sT = 1 / (sFs)
v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)
np.save(f'analysis/k_sweep/vx_{ts}.npy', vx)
#sQBit = 16
#sLsB = (2)/(2**sQBit)
#vx = (np.round(vx/sLsB))/(2**(sQBit-1))
################################################
##########  ########  #######  #################
vLPRangeFs          = [0, 0, sSigFmax, sSigFmax]
vLPfilterRad        = sg.freq2rad(vLPRangeFs, sFs)
vLPRangeFs          = [0, 0, sSigFmax, sSigFmax]
vLPfilterRadIdeal   = sg.freq2rad(vLPRangeFs, sFs)
vLPRangeFsDFT       = [0, 0, sSigFmax+1, sSigFmax+5]
vLPfilterRadDFT     = sg.freq2rad(vLPRangeFsDFT, sFs)
vBPRangeFs          = [180, 200, 311, 331]
vBPRangeRad         = sg.freq2rad(vBPRangeFs, sFs)
mOptFilterRange     = np.vstack((vLPfilterRad, vBPRangeRad))
vMaxVal             = [1, 0.5]
### Generate ideal matrices ###
vrIdeal, vHShift, vMask = filt.idealBinFilt(sNbins, vLPfilterRadIdeal, sValPass=1.0, sValStop=0.0, bFull=True)
mRIdeal = scLinAlg.toeplitz(vrIdeal)

sFpb        = sSigFmax
sFsb        = sFpb + 10
sApbdB      = 0.001
sAsbdB      = 50



######## Create Filter ########
(vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False)
np.save('saves/vWcoeff.npy', vWcoeff)

mOnes = np.ones((sNbins,sNbins))
mSigDeltaFilt = np.tril(mOnes)

vCoeffZ, mLobes = sa.detZC(vWcoeff, None)
vPruning, sPrunIdx  = sp.enLobePruning(vWcoeff, mLobes, 0.1, 8, True)
vWcoeffcut = vWcoeff[sPrunIdx+1::]

vx = mRIdeal @ vx
#### ISCAS 25 #### For comparison or as INIT value
vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQ(vx, vWcoeffcut, sBSize, 'grb')
# np.save('saves/vBSequBlock.npy', vBSequBlock)

# Quantize the input signal
_,_,vW = filt.idealBinFilt(sNbins, vLPfilterRadDFT, vMaxVal[0], 0.00001, True)
#vW_D = sp.getFiltWeights(vW, sK, vLPfilterRad)
    
for winIdx in range(sM//2-7, sM//2+13):
    
    vWDFT = abs(np.fft.fft(vWcoeff,1024))
    vBDFT, vBlockErr = obq.iterBlockQDFT(vx, vW, sK, sM, sL_max, vLPfilterRadDFT, sHop, winIdx, 'grb', True)
    #iterBlockQDFT(vx, vW, sK, sM, sL, mRange, sHop=None, sType='grb', verbose=True)
    
    # % % % % % % % % % % % % % % % %
    # EVALUATION
    # % % % % % % % % % % % % % % % %
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
    vDiffSingleMag = 20*sa.safelog10(np.abs(vDiffSingle) / np.max(abs(vX)))
    
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

    log_sweep_result(winIdx, vBlockErr,
                 sVB_SNRdbIdeal, sVBBlock_SNRdbIdeal,
                 sVB_SNRdb,      sVBBlock_SNRdb,
                 sM, sHop, sNbins, sK, 3*sM,
                 sSigFmin, sSigFmax, sL_max)

