# %%
# system packages
import numpy as np

#Plotting
import matplotlib.pyplot as mtplt
import matplotlib.gridspec as gridspec

#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import time

mtplt.rcParams['mathtext.fontset'] = 'stix'
mtplt.rcParams['font.family'] = 'STIXGeneral'

# %%
import sys,os
sys.path.append('../01_Library')
# individual packages
import sg, sa, sp, obq, filt, misc

mtplt.close('all')
np.seterr(all='ignore')

strBasePath = "Tests/TestSignals"
strDir      = os.path.join(strBasePath,"TestSigs_sF2048_Fmax115_N2048")
strTestFile = "Test_wCut_3.txt"
os.makedirs(strDir, exist_ok=True)

sNumTestSignals = 100

for sIdx in range(sNumTestSignals):
    print(f"Run {sIdx+1}/{sNumTestSignals}")
    strFile  = os.path.join(strDir, f"sig_{sIdx:03d}.npz")
    tSigData = np.load(strFile)
    
    sNbins   = 2048
    sFs      = tSigData["sFs"]
    vxFrequ  = tSigData["vxFrequ"]
    vxPhase  = tSigData["vxPhase"]
    sSigFmax = tSigData["sSigFmax"]
    
    #sBlockSize = 16 #32
    sM         = 32 #16
    sL_max     = sNbins
    sHop       = 16
    sK         = 115
    
    sBSize = 16
    
    sT = 1 / (sFs)
    v_n = np.arange(sNbins).reshape(-1, 1)
    vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
    vx = sg.MFnormalize(vx, -1, 1)
    s, J = sp.getBestBlockidx(vx,sM,sHop,5)
    
    ################################################
    ##########  ########  #######  #################
    vLPRangeFs          = [0, 0, sSigFmax, sSigFmax]
    vLPfilterRad        = sg.freq2rad(vLPRangeFs, sFs)
    vLPRangeFs          = [0, 0, sSigFmax, sSigFmax]
    vLPfilterRadIdeal   = sg.freq2rad(vLPRangeFs, sFs)
    vBPRangeFs          = [180, 200, 311, 331]
    vBPRangeRad         = sg.freq2rad(vBPRangeFs, sFs)
    mOptFilterRange     = np.vstack((vLPfilterRad, vBPRangeRad))
    vMaxVal             = [1, 0.5]
    ### Generate ideal matrices ###
    vRIdeal, vRIdealShift, vW = filt.idealBinFilt(sNbins, vLPfilterRadIdeal, 1.0, 0.0, True)
    mRIdeal = scLinAlg.toeplitz(vRIdeal)
    
    sFpb        = sSigFmax
    sFsb        = sFpb + 40
    sApbdB      = 0.001
    sAsbdB      = 60
    
    ######## Create Filter ########
    (vWcoeff, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.fir_calcLPKaiser(sFs, sFpb, sFsb, sApbdB, sAsbdB, None, False)
    np.save('saves/vWcoeff.npy', vWcoeff)
    
    mOnes = np.ones((sNbins,sNbins))
    mSigDeltaFilt = np.tril(mOnes)
    
    vCoeffZ, mLobes = sa.detZC(vWcoeff, None)
    vPruning, sPrunIdx  = sp.enLobePruning(vWcoeff, mLobes, 0.1, 8, True)
    vWcoeffcut = vWcoeff[sPrunIdx+1::]
    
    #### ISCAS 25 #### For comparison or as INIT value
    vBSequBlock, vEL2, vBlockIdx = obq.iterBlockQ(vx, vWcoeffcut, sBSize, 'grb')
    np.save('saves/vBSequBlock.npy', vBSequBlock)
    
    # Quantize the input signal
    vFiltIdeal, _ ,vW = filt.idealBinFilt(sNbins, vLPfilterRad, vMaxVal[0], 0, True)
    vW_D = sp.getFiltWeights(vW, sK, vLPfilterRad)
    
    vBDFT = obq.iterBlockQDFT(vx, vLPfilterRad, vW_D, sK, sM, sL_max, sHop, 'grb', True, 1)
    np.save('saves/vBDFT.npy', vBDFT)
    
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
    
    misc.saveScen2txt(vx, J, s, sM, sNbins, sHop, sFs, sSigFmax, round(sVB_SNRdbIdeal, 2), round(sVBBlock_SNRdbIdeal, 2), strTestFile)
    time.sleep(0.01)
