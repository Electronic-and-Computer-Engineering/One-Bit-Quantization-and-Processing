"""
Vollstaendige, korrekte OLA-Konstruktion:

- Drei identische Hann-Fenster der Breite W=70=2Hop+M+2Hop,
  zentriert auf M_0, M_1, M_2 (jeweils um Hop=10 verschoben).

- MEMORY-TERM E_l (fix, Index < Start(M_2)=20):
  sampleweise SUMME aller drei Glocken (M0+M1+M2, soweit >0)
  multipliziert mit dem dort bekannten d[n] = x[n]-b[n].
  (b[n] kommt aus M_0 fuer n<10, aus M_1 fuer n>=10 -- da sich
   M_0 und M_1 nur um Hop=10 unterscheiden und im Ueberlapp[10,30)
   beide "g\u00fcltig" waeren, nehmen wir an, dass das SIGNAL dort
   eindeutig ist -- x[n] ist ja eine einzige Zeitreihe, b[n] auch:
   jedes Sample hat genau EIN b, unabhaengig davon wie viele
   Bloecke es "gesehen" haben. Das loest die Ueberlapp-Frage von
   selbst: b[n] ist schlicht das (einzige, bereits final
   entschiedene) Bit an Position n.)

- GEWICHTUNG INNERHALB M_2 (Index >= 20, Variable):
  sampleweise SUMME aus M_0-Glocken-Auslauf + M_1-Glocken-Auslauf
  + M_2-eigene-Glocke (M_3/M_4 Auslauf weggelassen, wie besprochen).
"""

import numpy as np
np.random.seed(0)

Hop = 10
M = 30
W = 2*Hop + M + 2*Hop   # 70
K = 256

# Zeitachse: brauchen 0..49 (M_0,M_1,M_2 mit Hop=10 versetzt)
N = 50
n_full = np.arange(N)
freq_bins = [3,5,7]
x_full = np.zeros(N)
for fb in freq_bins:
    ph = np.random.uniform(0,2*np.pi)
    x_full += np.cos(2*np.pi*fb*n_full/N + ph)
x_full = x_full/np.max(np.abs(x_full))*0.8

M0_idx = np.arange(0,30)
M1_idx = np.arange(10,40)
M2_idx = np.arange(20,50)

# EIN b[n] pro Sample-Position, 0..39 (M0/M1-Bereich) -- final,
# eindeutig. Wir erzeugen es einfach sequentiell: erst M_0 entscheidet
# 0..29, dann "rutscht" die Optimierung weiter -- fuer dieses Experiment
# nehmen wir vereinfachend sign(x) als Stand-in fuer "bereits optimal
# entschieden", EINMAL ueber den ganzen Bereich, nicht doppelt.
b_known = np.sign(x_full[:40]); b_known[b_known==0]=1
d_known = x_full[:40] - b_known   # EIN eindeutiges d[n] pro Position

def hann_centered(n_axis, center, width):
    rel = (n_axis-center)/width + 0.5
    w = np.zeros_like(n_axis, dtype=float)
    inside = (rel>=0)&(rel<=1)
    w[inside] = 0.5-0.5*np.cos(2*np.pi*rel[inside])
    return w

center_M0 = 14.5
center_M1 = center_M0 + Hop     # 24.5
center_M2 = center_M1 + Hop     # 34.5

# Glocken jeweils ueber den GESAMTEN bekannten Bereich (0..39) ausgewertet
n_known = np.arange(40).astype(float)
w_M0_on_known = hann_centered(n_known, center_M0, W)
w_M1_on_known = hann_centered(n_known, center_M1, W)
w_M2_on_known = hann_centered(n_known, center_M2, W)   # M_2-Glocke reicht zurueck bis hier

# Sampleweise SUMME aller drei (das ist das OLA-Prinzip)
w_sum_known = w_M0_on_known + w_M1_on_known + w_M2_on_known

print("Gewichte je Glocke + Summe, ueber den bekannten Bereich (Index < 20 = vor M_2):")
for i in [0,5,9,10,15,19]:
    print(f"  n={i:2d}  M0={w_M0_on_known[i]:.3f}  M1={w_M1_on_known[i]:.3f}  M2={w_M2_on_known[i]:.3f}  SUMME={w_sum_known[i]:.3f}  d={d_known[i]:.3f}")

# Memory gilt NUR fuer Index < 20 (Start von M_2) -- harter Schnitt, wie festgelegt
mask_memory = n_known < 20
w_memory = np.where(mask_memory, w_sum_known, 0.0)

k = np.arange(K).reshape(-1,1)
F_known = np.exp(-1j*2*np.pi*k*n_known.reshape(1,-1)/N)
E_fix = (w_memory[None,:]*F_known) @ d_known

# --- Gewichtung INNERHALB M_2 (Index 20..49) ---
M2_idx_f = M2_idx.astype(float)
w_M0_on_M2 = hann_centered(M2_idx_f, center_M0, W)   # Auslauf von M0's Glocke in M2
w_M1_on_M2 = hann_centered(M2_idx_f, center_M1, W)   # Auslauf von M1's Glocke in M2
w_M2_on_M2 = hann_centered(M2_idx_f, center_M2, W)   # M2's eigene Glocke

w_total_M2 = w_M0_on_M2 + w_M1_on_M2 + w_M2_on_M2

print(f"\nGewichtung innerhalb M_2 (Summe aus M0-Auslauf+M1-Auslauf+M2-eigen):")
for i in [0,5,10,15,20,25,29]:
    print(f"  n_local={i:2d}  M0_auslauf={w_M0_on_M2[i]:.3f}  M1_auslauf={w_M1_on_M2[i]:.3f}  M2_eigen={w_M2_on_M2[i]:.3f}  SUMME={w_total_M2[i]:.3f}")

x_M2 = x_full[M2_idx]
F_M2 = np.exp(-1j*2*np.pi*k*M2_idx.reshape(1,-1)/N)

def solve(x_M, F_M, w_M, E_fix):
    Fw = w_M[None,:]*F_M
    b = np.sign(x_M); b[b==0]=1
    def cost(b):
        d = x_M-b
        spec = E_fix + Fw@d
        return np.sum(np.abs(spec)**2)
    cur=cost(b); improved=True
    while improved:
        improved=False
        for i in range(M):
            bt=b.copy(); bt[i]*=-1
            c=cost(bt)
            if c<cur-1e-12:
                b,cur=bt,c; improved=True
    return b,cur

b0,c0 = solve(x_M2, F_M2, np.ones(M), np.zeros(K,dtype=complex))     # Referenz
b1,c1 = solve(x_M2, F_M2, np.ones(M), E_fix)                          # nur Memory
b2,c2 = solve(x_M2, F_M2, w_total_M2, E_fix)                          # voll: Memory+OLA-Gewicht
b3,c3 = solve(x_M2, F_M2, w_total_M2, np.zeros(K,dtype=complex))      # nur OLA-Gewicht, kein Memory

d0,d1,d2,d3 = x_M2-b0, x_M2-b1, x_M2-b2, x_M2-b3

print(f"\n{'n':>3} {'x':>7} {'w_tot':>6} {'|d0|':>7} {'|d1|M':>7} {'|d2|MF':>7} {'|d3|F':>7}  flags")
for i in range(M):
    flag=""
    if b0[i]!=b1[i]: flag+=" M"
    if b0[i]!=b2[i]: flag+=" MF"
    if b0[i]!=b3[i]: flag+=" F"
    print(f"{i:3d} {x_M2[i]:7.3f} {w_total_M2[i]:6.3f} {abs(d0[i]):7.4f} {abs(d1[i]):7.4f} {abs(d2[i]):7.4f} {abs(d3[i]):7.4f}{flag}")

edge=list(range(5))+list(range(M-5,M)); center=list(range(5,M-5))
for name,d in [("Fall0 (nichts)",d0),("Fall1 (nur Memory)",d1),("Fall2 (Memory+OLA-Gewicht)",d2),("Fall3 (nur OLA-Gewicht)",d3)]:
    e=np.sum(d[edge]**2); c=np.sum(d[center]**2)
    print(f"{name:28s} Rand={e:7.4f} Mitte={c:7.4f} Rand-Anteil={100*e/(e+c):5.1f}%")

print(f"\nBits unterschiedlich zu Fall0:  M={np.sum(b0!=b1)}  MF={np.sum(b0!=b2)}  F={np.sum(b0!=b3)}  (von {M})")
print(f"Kosten: c0={c0:.4f} c1={c1:.4f} c2={c2:.4f} c3={c3:.4f}")