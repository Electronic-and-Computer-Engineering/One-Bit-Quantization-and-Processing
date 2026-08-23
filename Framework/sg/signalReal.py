import numpy as np

def signalReal(vK, sN, sBeta=0.0, vAmp=None, vPhase=None, bUseCos=True):
    """
    Real multi-tone digital test signal (rad/sample).

    vK              -> bins used Vekor resulting in ω_k = (2π/sN) * (vK + sBeta),  with 0 <= ω_k <= π
    sBeta           -> off-bin indicator (global fractional-bin offset) sBeta = 0 [onBin] sBeta != 0 [offBin] 
    sN              -> number of samples
    vAmp            -> Alternating amplitude factors, vAmp=None (all amps are 1)
    vPhase          -> Alternating phase factors, vPhase=None (φ_k ~ U[0, 2π)) 
    bUseCos         -> Use cosine oscillator bUseCose = True (cos), bUseCose = False (sin)              
     
    x[n] = Σ_k A_k * cos(ω_k n + φ_k)   (or sin if use_cos=False)

    Returns
    -------
    vSig    : (Ns,) ndarray
        Real signal
    vOmegaK : (K,) ndarray
        Digital frequencies (rad/sample)
    """
    # enforce vector shapes (1D)
    sN = int(sN)
    if sN <= 0:
        raise ValueError("sN must be a positive integer")
        
    vn = np.arange(sN, dtype=float)
    vK = np.asarray(vK, dtype=float).reshape(-1)
    sK = vK.size
    

    sBeta = float(sBeta)
    if not (0.0 <= sBeta < 1.0):
        raise ValueError("sBeta must satisfy 0 <= sBeta < 1")

    # ω_k and constraint to [0, π]
    vOmegaK = (2.0 * np.pi / sN) * (vK + sBeta)   # (K,)
    if np.any(vOmegaK < -1e-12) or np.any(vOmegaK > np.pi + 1e-12):
        raise ValueError("vK/sBeta violate 0 <= ω_k <= π")
    
    rng = np.random.default_rng()

    if vAmp is None:
        vAmp = np.ones(sK)
    elif isinstance(vAmp, str) and vAmp == "ampRnd":
        vAmp = rng.random(sK)
    else:
        vAmp = np.asarray(vAmp, dtype=float).reshape(-1)
        if vAmp.size != sK:
            raise ValueError("vAmp must match vK length")
    
    if vPhase is None:
        vPhase = 2.0 * np.pi * rng.random(sK)
    else:
        vPhase = np.asarray(vPhase, dtype=float).reshape(-1)
        if vPhase.size != sK:
            raise ValueError("vPhase must match vK length")

    # build (K,Ns) phase matrix via outer product
    mArg = np.outer(vOmegaK, vn) + vPhase[:, None]     # (K,Ns)
    mBasis = np.cos(mArg) if bUseCos else np.sin(mArg) # (K,Ns)

    vSig = (vAmp[:, None] * mBasis).sum(axis=0)        # (Ns,)

    return vSig, vOmegaK