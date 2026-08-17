import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === EINSTELLUNGEN ===
PARAM_DIR = ""  # ggf. anpassen
BLOCK_DIR = os.path.join(PARAM_DIR, "anBlock")


# === LADEN DER PARAMETER UND MATRIZEN ===
sL = int(np.load(os.path.join(PARAM_DIR, "sL.npy")))
sM = int(np.load(os.path.join(PARAM_DIR, "sM.npy")))
sK = int(np.load(os.path.join(PARAM_DIR, "sK.npy")))
vW = np.load(os.path.join(PARAM_DIR, "vW.npy"))
mF = np.load(os.path.join(PARAM_DIR, "mF.npy"))
sFs = np.load(os.path.join(PARAM_DIR, "sFs.npy"))
sHop = np.load(os.path.join(PARAM_DIR, "sHop.npy"))
mFW = np.diag(vW) @ mF
Q = sL + sM
N_BLOCKS = len([f for f in os.listdir(BLOCK_DIR) if f.endswith(".npz")])

fs = sFs  # Samplingrate

# === PLOT-FENSTER OBJEKTE
fig, axs = None, None
lines = {}

def load_block(idx):
    path = os.path.join(BLOCK_DIR, f"block_{idx:03d}.npz")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    else:
        print(f"[❌] Block {idx:03d} nicht gefunden.")
        return None

def dbnorm(x):
    x = np.abs(x)
    x = x / np.max(x) if np.max(x) > 0 else x
    return 20 * np.log10(np.maximum(x, 1e-12))

def plot_block(idx, data, data_prev=None):
    global fig, axs, lines

    x_block = data["x_block"]
    b_curr = data["b_full"]
    b_prev = data_prev["b_full"] if data_prev else np.zeros_like(b_curr)
    block_err = data["vBlockError"]
    meanX     = data["vxMean"]

    X_target = mFW @ x_block
    X_hat = mFW @ b_curr
    X_error = X_target - X_hat
    freqs = np.linspace(0, fs, sK)

    X_target_db = dbnorm(X_target)
    X_hat_db = dbnorm(X_hat)
    X_error_db = dbnorm(X_error)

    if fig is None:
        fig, axs = plt.subplots(5, 1, figsize=(16, 10), constrained_layout=True)

        # (1) b_prev
        axs[0].set_ylim([-1.1, 1.1])
        axs[0].set_ylabel("b_prev")
        lines["b_prev"], = axs[0].step(np.arange(len(b_prev)), b_prev, where="mid")
        lines["box_prev"] = axs[0].add_patch(
            patches.Rectangle((sHop, -1), sL-1, 2, color="lightgreen", alpha=0.3)
        )

        # (2) b_curr
        axs[1].set_ylim([-1.1, 1.1])
        axs[1].set_ylabel("b_curr")
        lines["b_curr"], = axs[1].step(np.arange(len(b_curr)), b_curr, where="mid")
        lines["box_curr"] = axs[1].add_patch(
            patches.Rectangle((0, -1), sL-1, 2, color="lightgreen", alpha=0.3)
        )

        # (3) Spektrum
        axs[2].set_xlim([0, fs / 2])
        axs[2].set_xlabel("Frequency [Hz]")
        axs[2].set_ylabel("Magnitude [dB]")
        axs[2].grid(True)
        lines["X_target"], = axs[2].plot(freqs, X_target_db, label="|X_target| [dB]")
        lines["X_hat"],    = axs[2].plot(freqs, X_hat_db, label="|X_hat| [dB]")
        lines["X_error"],  = axs[2].plot(freqs, X_error_db, label="|X_error| [dB]", linestyle="--")

        axs[2].legend()

        # (4) Fehlerverlauf
        axs[3].set_xlabel("Iteration")
        axs[3].set_ylabel("Blockfehler")
        axs[3].set_title("vBlockError Verlauf")
        axs[3].grid(True)
        lines["err"], = axs[3].plot(np.arange(len(block_err)), block_err, marker='o')

        
        axs[4].set_xlabel("Iteration")
        axs[4].set_ylabel("Mean")
        axs[4].set_title("vxMean Verlauf")
        axs[4].grid(True)
        lines["xMean"], = axs[4].plot(np.arange(len(meanX)), abs(meanX), marker='o')

        fig.suptitle(f"Block {idx:03d}")
        plt.show(block=False)
        plt.pause(0.1)

    else:
        # (1) b_prev
        lines["b_prev"].set_data(np.arange(len(b_prev)), b_prev)
        lines["box_prev"].set_xy((sM, -1))

        # (2) b_curr
        lines["b_curr"].set_data(np.arange(len(b_curr)), b_curr)
        lines["box_curr"].set_xy((0, -1))

        # (3) dB-Spektren
        lines["X_target"].set_ydata(X_target_db)
        lines["X_hat"].set_ydata(X_hat_db)
        lines["X_error"].set_ydata(X_error_db)
        axs[2].relim()
        axs[2].autoscale_view()

        # (4) Fehlerverlauf
        lines["err"].set_data(np.arange(len(block_err)), block_err)
        axs[3].set_xlim([0, len(block_err)])
        axs[3].set_ylim([0, np.max(block_err)*1.05 if len(block_err) > 0 else 1])
        
        # (5) Fehlerverlauf
        lines["xMean"].set_data(np.arange(len(meanX)), meanX)
        axs[4].set_xlim([0, len(meanX)])
        axs[4].set_ylim([0, np.max(meanX)*1.05 if len(meanX) > 0 else 1])

        fig.suptitle(f"Block {idx:03d}")
        fig.canvas.draw_idle()
        plt.pause(0.1)

# === INTERAKTIVE EINGABE ===
while True:
    cmd = input(f"\n🔢 Blocknummer eingeben (0–{N_BLOCKS-1}, q zum Beenden): ").strip().lower()
    if cmd == "q":
        print("🛑 Analyse beendet.")
        break

    try:
        idx = int(cmd)
        if 0 <= idx < N_BLOCKS:
            data = load_block(idx)
            data_prev = load_block(idx - 1) if idx > 0 else None
            if data:
                plot_block(idx, data, data_prev)
        else:
            print(f"[⚠️] Bitte gib eine Zahl zwischen 0 und {N_BLOCKS-1} ein.")
    except ValueError:
        print("[❌] Ungültige Eingabe. Nur Zahlen oder 'q' erlaubt.")
