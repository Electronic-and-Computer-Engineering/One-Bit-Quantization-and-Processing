import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../01_Library')
# individual packages
import sg, sa, sp, obq, filt 

def dftMat(N, K, range=None, normalize=False, unit='rad'):
    """
    Constructs a complex-valued DFT matrix (K x N) for full or partial spectrum analysis.

    Parameters:
    ----------
    N : int
        Length of time-domain signal (number of columns).
    K : int
        Number of frequency bins (number of rows).
    range : array-like of shape (2,), optional
        Frequency interval [low, high].
        If None: uses full range [0, 2π) (in rad/sample).
    unit : str, optional
        Unit of the range: 'rad' (radians, default) or 'f' (normalized frequency in [0, 1]).
    normalize : bool, optional
        If True, applies 1/√N normalization (Parseval-compatible).

    Returns:
    -------
    F : ndarray of shape (K, N)
        Complex-valued DFT matrix.
    omega_k : ndarray of shape (K,)
        Frequency grid (in rad/sample) corresponding to the rows of F.
    """

    n = np.arange(N)  # time indices

    if range is None:
        # Full-band DFT (uniformly spaced over [0, 2π))
        omega_k = 2 * np.pi * np.arange(K) / K
    else:
        range = np.asarray(range)
        if range.shape != (2,):
            raise ValueError("range must be a 2-element array-like: [low, high]")

        if unit == 'f':
            range = 2 * np.pi * range
        elif unit != 'rad':
            raise ValueError("unit must be either 'rad' or 'f'")

        omega_k = np.linspace(range[0], range[1], K)

    F = np.exp(-1j * np.outer(omega_k, n))  # shape: (K, N)

    if normalize:
        F /= np.sqrt(N)

    return F, omega_k

# Signalparameter
sFs = 128
sNbins = 4096
sBSize = 64
sHop = 4
sSigFmax = 50
sK = 512

vLPRangeFs      = [5, sSigFmax]
vLPfilterRad    = sg.freq2rad(vLPRangeFs, sFs)
vMaxVal         = [1.0, 0.0]

# Generiere Signal
vxFrequ         = (np.arange(0, sSigFmax, step=2)).reshape(-1, 1)
vxPhase         = np.random.rand(len(vxFrequ), 1) * 2 * np.pi
v_n             = np.arange(sNbins).reshape(-1, 1)
vx, vTime       = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx              = sg.MFnormalize(vx, -1, 1).flatten()

# Zero Padding am Anfang
vx_padded       = np.concatenate([np.zeros(sBSize), vx])
nBlocks         = (len(vx_padded) - sBSize) // sHop

# DFT

mDFT,_ = dftMat(sBSize, sK, vLPfilterRad)
vFreq = np.fft.fftfreq(sK, d=1/sFs)

# Plot Setup
block_idx = [0]
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, sFs / 2)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Frequenz [Hz]")
ax.set_ylabel("Betrag der DFT")
ax.set_title("Leertaste für nächsten Block – ESC zum Beenden")

def plot_next_block(event):
    if event.key == 'escape':
        plt.close(fig)
        return
    if event.key != ' ':
        return

    idx = block_idx[0]
    sStart = idx * sHop
    sEnd = sStart + sBSize
    if sEnd > len(vx_padded):
        print("Ende erreicht.")
        plt.close(fig)
        return

    block = vx_padded[sStart:sEnd]
    X = mDFT @ block
    X_mag = np.abs(X) / np.max(np.abs(X))

    line.set_data(vFreq, X_mag)
    ax.set_title(f"Block {sStart}:{sEnd} (DFT via Matrix)")
    fig.canvas.draw()
    block_idx[0] += 1

fig.canvas.mpl_connect('key_press_event', plot_next_block)
plt.show()

