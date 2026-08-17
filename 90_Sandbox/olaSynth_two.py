"""
Overlap-Add (OLA) — interaktive Schritt-für-Schritt Visualisierung
===================================================================

Steuerung (Plotfenster muss fokussiert sein):
    LEERTASTE   -> nächstes Fenster hinzufügen (Abstand = hop)
    LINKS-PFEIL -> ein Fenster zurück
    r           -> Reset (zurück zu 1 Fenster)
    q           -> Fenster schließen

Es wird:
  - oben jedes einzelne (verschobene) Fenster gezeigt
  - unten die NORMIERTE OLA-Summe (Summe / max(Summe)) gezeigt
  - eine blaue Zone der Breite M, die bei Index 0 startet und bei
    jedem neuen Fenster um `hop` weiterwandert, in beiden Plots
    hervorgehoben

Fenstertyp ist austauschbar über `scipy.signal.get_window`.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window


# ----------------------------------------------------------------------
# 1) Parameter (hier anpassen)
# ----------------------------------------------------------------------

N = 1024             # Gesamtlänge des Signals
winLen = 48          # Fensterlänge
hop = 8              # Hop-Size (Schrittweite)
M = 32                # Breite der blauen Zone (unabhängig von winLen/hop)
window_type = "hann"   # 'hann', 'hamming', 'blackman', 'boxcar', 'bartlett', ...


# ----------------------------------------------------------------------
# 2) Fenster erzeugen
# ----------------------------------------------------------------------

def make_window(win_type: str, win_len: int) -> np.ndarray:
    return get_window(win_type, win_len, fftbins=False)


win = make_window(window_type, winLen)
max_windows = max(1, (N - winLen) // hop + 1) if winLen <= N else 1


# ----------------------------------------------------------------------
# 3) State + Berechnung der OLA-Summe für die aktuelle Schrittzahl
# ----------------------------------------------------------------------

state = {"step": 1}  # Anzahl aktiver Fenster, beginnt bei 1


def compute_ola(step: int):
    """Berechnet Start-Indizes, rohe OLA-Summe und normierte OLA-Summe
    für `step` aktive Fenster.
    """
    starts = [i * hop for i in range(step)]
    ola_sum = np.zeros(N)
    for s in starts:
        end = min(s + winLen, N)
        ola_sum[s:end] += win[: end - s]
    max_val = ola_sum.max() if ola_sum.max() > 0 else 1.0
    ola_norm = ola_sum / max_val
    return starts, ola_sum, ola_norm, max_val


# ----------------------------------------------------------------------
# 4) Plot-Setup
# ----------------------------------------------------------------------

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.suptitle("Overlap-Add | →: nächstes Fenster | ←: zurück | r: Reset",
             fontsize=11)

window_lines = []          # aktuell gezeichnete Fensterkurven (oben)
zone_patches = []          # blaue Zonen-Rechtecke (oben + unten)
sum_line, = ax_bottom.plot([], [], color="black", linewidth=2, label="Norm. OLA-Summe")
ref_line = ax_bottom.axhline(1.0, color="red", linestyle="--", linewidth=1,
                              alpha=0.6, label="1.0 (Maximum)")

ax_top.set_xlim(0, N)
ax_top.set_ylim(-0.05, 1.15)
ax_top.set_ylabel("Amplitude")
ax_top.grid(True, alpha=0.3)

ax_bottom.set_xlim(0, N)
ax_bottom.set_ylim(-0.05, 1.15)
ax_bottom.set_xlabel("Sample-Index n")
ax_bottom.set_ylabel("Σ / max(Σ)")
ax_bottom.grid(True, alpha=0.3)
ax_bottom.legend(loc="upper right", fontsize=9)

colors = plt.cm.tab10.colors


def redraw():
    """Zeichnet den kompletten Plot für state['step'] neu."""
    step = state["step"]
    starts, ola_sum, ola_norm, max_val = compute_ola(step)

    # --- alte Elemente entfernen ---
    for ln in window_lines:
        ln.remove()
    window_lines.clear()
    for p in zone_patches:
        p.remove()
    zone_patches.clear()

    # --- einzelne Fenster (oben) zeichnen ---
    for i, s in enumerate(starts):
        t = np.arange(s, min(s + winLen, N))
        y = win[: len(t)]
        is_last = (i == len(starts) - 1)
        ln, = ax_top.plot(
            t, y,
            color=colors[i % len(colors)],
            linewidth=2.2 if is_last else 1.1,
            alpha=1.0 if is_last else 0.5,
        )
        window_lines.append(ln)

    # --- OLA-Summe (unten) aktualisieren ---
    sum_line.set_data(np.arange(N), ola_norm)

    # --- blaue M-Zone: Start wandert mit dem letzten Fenster ---
    zone_start = starts[-1]
    zone_width = min(M, N - zone_start)
    for ax in (ax_top, ax_bottom):
        rect = ax.axvspan(zone_start, zone_start + zone_width,
                           color="tab:blue", alpha=0.18, zorder=0)
        zone_patches.append(rect)

    ax_top.set_title(
        f"{window_type}-Fenster — aktiv: {step}/{max_windows}   "
        f"(letztes Fenster bei n={zone_start})"
    )
    ax_bottom.set_title(
        f"Normierte OLA-Summe (roh. max={max_val:.3f})   "
        f"blaue Zone M={M} bei n=[{zone_start}, {zone_start + zone_width})"
    )

    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "right":
        if state["step"] < max_windows:
            state["step"] += 1
            redraw()
    elif event.key == "left":
        if state["step"] > 1:
            state["step"] -= 1
            redraw()
    elif event.key == "r":
        state["step"] = 1
        redraw()
    elif event.key == "q":
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)

# Erstes Fenster initial zeichnen
redraw()

plt.tight_layout()
plt.show()