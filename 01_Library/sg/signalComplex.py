import numpy as np

def signalComplex(vK, sN, sBeta=0.0, vAmp=None, vPhase=None):
    """
    Complex multi-tone digital test signal (rad/sample).

    vK      -> bin index vector, yielding  ω_k = (2π/sN) * (vK + sBeta)
    sBeta   -> global off-bin offset: sBeta=0 [on-bin], sBeta!=0 [off-bin]
    sN      -> number of samples (signal length)
    vAmp    -> amplitude factors:
               - vAmp=None      -> all amplitudes = 1
               - vAmp="ampRnd"  -> random amplitudes in [0,1]
               - vAmp=vector    -> user-provided amplitudes
    vPhase  -> phase factors (rad):
               - vPhase=None    -> random phases φ_k ~ U[0,2π)
               - vPhase=vector  -> user-provided phases

    Signal model:
        x[n] = Σ_k A_k * exp(j(ω_k n + φ_k))

    Returns
    -------
    vSig    : (sN,) ndarray (complex)
        Complex signal
    vOmegaK : (K,) ndarray
        Digital frequencies (rad/sample)
    """
    # enforce vector shapes (1D)
    vn = np.arange(sN, dtype=float).reshape(-1)           # (sN,)
    vK = np.asarray(vK, dtype=float).reshape(-1)          # (K,)

    sK = vK.size
    sN = int(sN)
    if sN <= 0:
        raise ValueError("sN must be a positive integer")

    sBeta = float(sBeta)
    if not (0.0 <= sBeta < 1.0):
        raise ValueError("sBeta must satisfy 0 <= sBeta < 1")

    # ω_k and constraint to [0, π] (keep same convention as signalReal)
    vOmegaK = (2.0 * np.pi / sN) * (vK + sBeta)           # (K,)
    if np.any(vOmegaK < -1e-12) or np.any(vOmegaK > np.pi + 1e-12):
        raise ValueError("vK/sBeta violate 0 <= ω_k <= π")

    # amplitudes (3 cases)
    if vAmp is None:
        vAmp = np.ones(sK)

    elif vAmp == "ampRnd":
        vAmp = np.random.rand(sK)

    else:
        vAmp = np.array(vAmp, dtype=float).reshape(-1)
        if vAmp.size != sK:
            raise ValueError("vAmp must match vK length")

    # phases
    if vPhase is None:
        rng = np.random.default_rng()
        vPhase = 2.0 * np.pi * rng.random(sK)
    else:
        vPhase = np.asarray(vPhase, dtype=float).reshape(-1)
        if vPhase.size != sK:
            raise ValueError("vPhase must match vK length")

    # build (K,N) phase matrix and synthesize complex exponentials
    mArg = np.outer(vOmegaK, vn) + vPhase[:, None]        # (K,sN)
    mBasis = np.exp(1j * mArg)                            # (K,sN)

    vSig = (vAmp[:, None] * mBasis).sum(axis=0)           # (sN,)

    return vSig, vOmegaK