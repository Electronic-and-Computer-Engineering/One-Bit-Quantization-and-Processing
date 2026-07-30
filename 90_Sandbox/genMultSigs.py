# %%
# system packages
import numpy as np

# %%
import os

sNbins    = 2048
sFs       = 2048
sSigFmax  = 115
vxFrequ   = (np.arange(0, sSigFmax, step=1)).reshape(-1, 1)

# --- Save only the time signal ---
strBasePath = "Tests/TestSignals"
strDir = os.path.join(strBasePath,f"TestSigs_sF{sFs}_Fmax{sSigFmax}_N{sNbins}")
os.makedirs(strDir, exist_ok=True)
sNumTestSignals = 200

for sIdx in range(sNumTestSignals):
    strFile = os.path.join(strDir, f"sig_{sIdx:03d}.npz")
    vxPhase     = np.random.rand(len(vxFrequ), 1) * 2 * np.pi   
    np.savez(strFile,
         vxFrequ=vxFrequ,
         vxPhase=vxPhase,
         sSigFmax=sSigFmax,
         sFs=sFs)
    
    

