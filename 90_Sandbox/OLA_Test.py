import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import sys
sys.path.append('../01_Library')
import sg, sa

# === Signal laden und normalisieren ===
vxFrequ = np.load('saves/vxFrequ.npy')
vxPhase = np.load('saves/vxPhase.npy')
sSigFmax = np.load('saves/sSigFmax.npy')
sFs = np.load('saves/sFs.npy')

sNbins = 1024
v_n = np.arange(sNbins).reshape(-1, 1)
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhase, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1).squeeze()
vTime = vTime.squeeze()

# === Parameter ===
sM = 32
sH = sM // 2
sPad = sM // 2
window_type = 'hann'  # universell austauschbar
ApplyWindow = True

# === Analyse ===
mBlocks, vWin, vx_pad = sa.OLA_analysis(vx, sM, sH, window_type, sPad, ApplyWindow)
numBlocks = mBlocks.shape[1]
vTime_pad = np.linspace(0, (len(vx_pad)-1)/sFs, len(vx_pad))

# === Synthese ===
vx_hat = sa.OLA_synth(mBlocks, sM, sH, window_type, sPad, ApplyDePadding=True)

# === Bewertung ===
err = vx_hat - vx
mae = np.max(np.abs(err))
snr = 10 * np.log10(np.sum(vx**2) / (np.sum(err**2) + 1e-30))

# === Plot ===
fig, axs = plt.subplots(4, 1, figsize=(10, 10))

# (1) Original vs Rekonstruktion
axs[0].plot(vTime, vx, color='black', lw=1, label='Original')
axs[0].plot(vTime, vx_hat, color='red', lw=1, label='OLA-Rekonstruktion')
axs[0].legend()
axs[0].set_title("1) Original vs. OLA-Rekonstruktion")
axs[0].set_xlim(0, vTime[-1])
axs[0].set_ylabel("Amplitude")

# (2) Gefensterte Blöcke + Rekonstruktion
for p in range(numBlocks):
    start = p * sH
    ts = vTime_pad[start:start + sM]
    if len(ts) != sM:
        continue
    axs[1].plot(ts, mBlocks[:, p], color='gray', alpha=0.4)
axs[1].set_title("2) Gefensterte Blöcke + OLA-Ergebnis (rot)")
axs[1].set_xlim(0, vTime[-1])
axs[1].set_ylabel("Amplitude")

# (3) Einzelner Block + Mittelwerte
block_idx = min(3, numBlocks - 1)
start = block_idx * sH
ts = vTime_pad[start*4:start*4 + sM]
block = vx_pad[start*4:start*4 + sM]
block_w = vWin[:len(block)] * block
mean_unwin = np.mean(block)
mean_win = np.mean(block_w)

axs[2].plot(ts, block, color='gray', label=f'Ungefenstert (μ={mean_unwin:.3e})')
axs[2].plot(ts, block_w, color='blue', label=f'Gefenstert (μ={mean_win:.3e})')
axs[2].axhline(mean_unwin, color='gray', linestyle='--', lw=0.8)
axs[2].axhline(mean_win, color='blue', linestyle='--', lw=0.8)
axs[2].legend(loc='upper right')
axs[2].set_title(f"3) Block {block_idx}: Fensterwirkung")
axs[2].set_xlim(ts[0], ts[-1])
axs[2].set_xlabel("Zeit [s]")
axs[2].set_ylabel("Amplitude")

# (4) Summe aller Fensterfunktionen (OLA-Bedingung)
vWin = sp.get_window(window_type, sM, fftbins=False)
num_shifts = int(np.ceil(sM / sH)) + 2
sum_len = sM + sH * (num_shifts - 1)
vSum = np.zeros(sum_len)

for p in range(num_shifts):
    start = p * sH
    vSum[start:start + sM] += vWin

v_n = np.arange(len(vSum))
axs[3].plot(v_n, vSum, color='blue', label=r'$\sum_p w[n - pH]$')
axs[3].axhline(1, color='red', linestyle='--', label='Idealwert = 1')
axs[3].set_title(f"4) Summe der überlappenden Fenster ({window_type}, M={sM}, Hop={sH})")
axs[3].set_xlabel("Sampleindex n")
axs[3].set_ylabel("Σ Fensteramplitude")
axs[3].legend()
axs[3].grid(True)

fig.suptitle(f"OLA-Beweis mit Fenster '{window_type}': max|err|={mae:.2e}, SNR={snr:.1f} dB",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()