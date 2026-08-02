# system packages
import numpy as np

#Plotting
import matplotlib.pyplot as mtplt
import matplotlib.gridspec as gridspec

#Linear Algebraic, signal processing
#import scipy.linalg as scLinAlg
#import scipy.signal as sigP

# %%
# individual packages
import os
# individual packages created
# We would add more if neccessary
import sg, sa, filt, fStg

vCases = [
    {"strSig":"real","vAmp":None,"vPhase":None,"bUseCos":True,"sBatchSize":2000,"sN":2048,"sM":32,"sL":213,"sBeta":0.0,"mWD":np.array([[np.pi/180,np.pi/10],[3*np.pi/10,4*np.pi/10]]),
     "mR":np.array([[0.0,np.pi/10]]),"sBound":1.0,"kaiser":{"sApb":1.0,"sAsb":80.0,"sDeltaW":np.pi/180,"bMinPhase":False}},
    
    {"strSig":"real","vAmp":"ampRnd","vPhase":None,"bUseCos":True,"sBatchSize":100,"sN":2048,"sM":32,"sL":213,"sBeta":0.5,"mWD":np.array([[np.pi/180,np.pi/10],[3*np.pi/10,4*np.pi/10]]),
     "mR":np.array([[0.0,np.pi/10]]),"sBound":1.0,"kaiser":{"sApb":1.0,"sAsb":80.0,"sDeltaW":np.pi/180,"bMinPhase":False}}
        ]

os.makedirs("TestBatches", exist_ok=True)

for dictCase in vCases:

    caseDirName  = fStg.makeCaseName(dictCase)

    ## SETTINGS
    #
    #
    sBatchSize   = dictCase["sBatchSize"]
    sN           = dictCase["sN"]
    sL           = dictCase["sL"]
    sBeta        = dictCase["sBeta"]
    sBound       = dictCase["sBound"]
    strSig       = dictCase["strSig"]
    mWD          = dictCase["mWD"]    #provide /omega_{min} and /omega_{max} zones, each row indicates new zone
    mR           = dictCase["mR"]
    sM           = dictCase["sM"]
    dictKaiser   = dictCase["kaiser"]
    vAmp         = dictCase["vAmp"]
    vPhase       = dictCase["vPhase"]
    bUseCos      = dictCase["bUseCos"]
    bMinPhase    = dictKaiser["bMinPhase"]

    vK           = sg.getKFromWD(mWD,sN)

    mx = np.zeros((sN, sBatchSize), dtype=complex if strSig == "complex" else float)

    # %% Filter design -- constant within a case, therefore outside the batch loop

    # non-ideal FIRs (Kaiser)
    (vw, vOmegaW, vHrespW, sRpbW, sRsbW, sHpbMinW, sHpbMaxW, sHsbMaxW) = filt.fir_calcMBKaiser(
        mWD=mWD,
        sApb=dictKaiser["sApb"],
        sAsb=dictKaiser["sAsb"],
        sDeltaW=dictKaiser["sDeltaW"],   # width
        sTaps=sL,
        sMinPhase=bMinPhase
    )

    (vr, vOmegaR, vHrespR, sRpbR, sRsbR, sHpbMinR, sHpbMaxR, sHsbMaxR) = filt.fir_calcMBKaiser(
        mWD=mR,
        sApb=dictKaiser["sApb"],
        sAsb=dictKaiser["sAsb"],
        sDeltaW=dictKaiser["sDeltaW"],   # width
        sTaps=sL,
        sMinPhase=bMinPhase
    )

    # ideal FFT mask
    vwIdeal, vHShiftW, vMaskW = filt.idealBinFiltFromMW(sN, mWD, sValPass=1.0, sValStop=0.0, bFull=True)
    vrIdeal, vHShiftR, vMaskR = filt.idealBinFiltFromMW(sN, mR,  sValPass=1.0, sValStop=0.0, bFull=True)

    # %% Generation
    fnSignal = sg.signalComplex if strSig == "complex" else sg.signalReal

    for idxBatch in range(sBatchSize):

        vx, _   = fnSignal(vK, sN, sBeta, vAmp, vPhase, bUseCos)          # Create Signal
        vx      = sg.boundRange(vx, sBound)        # Bound the Signal to given range

        mx[:,idxBatch] = vx

    # %% Save
    np.savez(
    os.path.join("TestBatches", caseDirName + ".npz"),
    mx          = mx,
    vw          = vw,
    vr          = vr,
    vwIdeal     = vwIdeal,
    vrIdeal     = vrIdeal,
    sM          = sM,
    bMinPhase   = bMinPhase,
    )
        
    fStg.writeCaseMarkdown("TestBatches", caseDirName, dictCase)    

# Proof plot after patch idea
vX           = np.fft.fft(vx, sN)
vXMag        = 20 * sa.safelog10(np.abs(vX) / np.max(abs(vX)))
vR           = np.fft.fft(vw, sN)
vRMag        = 20 * sa.safelog10(np.abs(vR) / np.max(abs(vR)))
vW           = np.fft.fft(vwIdeal, sN)
vWMag        = 20 * sa.safelog10(np.abs(vW) / np.max(abs(vW)))

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
Pltgs = gridspec.GridSpec(4,1)

pltDiscTime = figOne.add_subplot(Pltgs[0,:])
pltDiscTime.plot(vx)
pltDiscTime.set_title('Input Signal')
pltDiscTime.set_xlabel('Samples $n$', fontsize = 11)
pltDiscTime.set_ylabel('Amplitude', fontsize = 11)
pltDiscTime.set_xlim([0,sN])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreq = figOne.add_subplot(Pltgs[1,:])
pltFreq.plot(vNormFrequ, vXMag)
#pltFreq.set_title('Input Signal')
pltFreq.set_xlabel('normalized Frequency', fontsize = 11)
pltFreq.set_ylabel('Magnitude (dB)', fontsize = 11)
pltFreq.set_xlim([0,2*np.pi])
pltFreq.set_xticks(xticks)
pltFreq.set_xticklabels(xtick_labels, fontsize=13)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreq = figOne.add_subplot(Pltgs[2,:])
pltFreq.plot(vNormFrequ, vWMag)
#pltFreq.set_title('Input Signal')
pltFreq.set_xlabel('normalized Frequency', fontsize = 11)
pltFreq.set_ylabel('Magnitude (dB)', fontsize = 11)
pltFreq.set_xlim([0,2*np.pi])
pltFreq.set_xticks(xticks)
pltFreq.set_xticklabels(xtick_labels, fontsize=13)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreq = figOne.add_subplot(Pltgs[3,:])
pltFreq.plot(vNormFrequ, vRMag)
#pltFreq.set_title('Input Signal')
pltFreq.set_xlabel('normalized Frequency', fontsize = 11)
pltFreq.set_ylabel('Magnitude (dB)', fontsize = 11)
pltFreq.set_xlim([0,2*np.pi])
pltFreq.set_xticks(xticks)
pltFreq.set_xticklabels(xtick_labels, fontsize=13)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')