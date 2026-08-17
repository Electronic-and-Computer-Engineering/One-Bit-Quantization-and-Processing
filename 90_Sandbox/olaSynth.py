import numpy as np
import matplotlib.pyplot as plt


def compute_ola_window(M, winLen, Hop, window_fn=np.hanning, normalize=False):
    """
    Computes the OLA weight function for a block of length M.
    One window is placed centered on M, then ov copies are placed
    to the left and right, each shifted by Hop samples.

    Works correctly for both even and odd winLen.

    Args:
        M          : block length in samples
        winLen     : window length in samples (even or odd)
        Hop        : hop size in samples
        window_fn  : window function, e.g. np.hanning, np.hamming, np.blackman
        normalize  : if True, normalize OLA sum to peak = 1

    Returns:
        ola_sum     : full OLA weight array (length = total_len)
        block_start : index of first sample of M in ola_sum
        ov          : number of shifted copies per side
    """
    ov        = int(np.floor(((winLen - M) / 2 + M) / Hop)) - 1
    half_win  = winLen // 2

    block_center = ov * Hop + half_win
    block_start  = block_center - M // 2
    total_len    = block_center + ov * Hop + half_win + (1 if winLen % 2 == 0 else 1)

    w_base  = window_fn(winLen)
    ola_sum = np.zeros(total_len)

    for i in range(0, 3):
        center = block_center + i * Hop
        s      = center - half_win
        e      = s + winLen
        ola_sum[s:e] += w_base

    if normalize:
        peak = np.max(ola_sum)
        if peak > 0:
            ola_sum /= peak

    return ola_sum, block_start, ov


# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
M         = 32
winLen    = 128   # odd
Hop       = 32
window_fn = np.hanning
normalize = True

# ------------------------------------------------------------------
# Compute OLA
# ------------------------------------------------------------------
ola_sum, block_start, ov = compute_ola_window(M, winLen, Hop, window_fn, normalize)

print(f"M={M}, winLen={winLen}, Hop={Hop}  ->  ov={ov}")
print(f"block=[{block_start}, {block_start+M}),  total_len={len(ola_sum)}")

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
half_win     = winLen // 2
block_center = block_start + M // 2
total_len    = len(ola_sum)
n            = np.arange(total_len)
w_base       = window_fn(winLen)

colors_past   = plt.cm.Greens(np.linspace(0.4, 0.9, ov))
colors_future = plt.cm.Reds(np.linspace(0.4, 0.9, ov))

fig, ax = plt.subplots(figsize=(14, 5))

# Individual windows
for i in range(0, 3):
    center = block_center + i * Hop
    s      = center - half_win
    e      = s + winLen

    w_full      = np.zeros(total_len)
    w_full[s:e] = w_base

    if i < 0:
        col = colors_past[abs(i) - 1]
        lbl = f'past copy    i={i:+d}  (center={center})'
    elif i > 0:
        col = colors_future[i - 1]
        lbl = f'future copy  i={i:+d}  (center={center})'
    else:
        col = 'steelblue'
        lbl = f'center copy  i= 0   (center={center})'

    ax.plot(n, w_full, color=col, lw=1.0, alpha=0.6, label=lbl)

# Full OLA sum (black)
ax.plot(n, ola_sum, 'k-', lw=2.5, label='OLA sum (normalized)', zorder=5)

# OLA sum up to end of block M (red)
ola_block                   = np.zeros(total_len)
ola_block[:block_start + M+1] = ola_sum[:block_start + M+1]
ax.plot(n, ola_block, 'r-', lw=2.5, label='OLA sum up to end of M', zorder=6)

# Block boundaries
ax.axvline(block_start,       color='blue', lw=1.5, ls='--')
ax.axvline(block_start + M,   color='blue', lw=1.5, ls='--', label='Block M')
ax.axvline(block_center,      color='blue', lw=1.0, ls=':',  label='Block center')
ax.axvspan(block_start, block_start + M, alpha=0.10, color='blue')
ax.axhline(1.0, color='gray', lw=1.0, ls=':', label='normalized peak = 1')

ax.set_title(f'OLA sum  |  M={M}, winLen={winLen}, Hop={Hop}, ov={ov}, '
             f'norm={normalize}, win={window_fn.__name__}')
ax.set_xlabel('Sample n')
ax.set_ylabel('Weight (normalized)')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()