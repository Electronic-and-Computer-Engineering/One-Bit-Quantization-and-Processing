import numpy as np
from scipy.signal import convolve

# Filter (w_0 ... w_8) und Signal (x_0 ... x_11)
w = [f"w_{i}" for i in range(9)]
M = 4
blocks = [
    [f"x_{i}" for i in range(0, 4)],
    [f"x_{i}" for i in range(4, 8)],
    [f"x_{i}" for i in range(8, 12)]
]

# Hilfsfunktion: Blockmatrix W^(k)
def block_matrix_terms(w, k, M):
    L = len(w)
    Wk = []
    for i in range(M):
        row = []
        for j in range(M):
            idx = k*M + i - j
            if 0 <= idx < L:
                row.append(f"{w[idx]}")
            else:
                row.append("0")
        Wk.append(row)
    return Wk

# Hilfsfunktion: Matrix-Vektor Multiplikation in Terme
def matvec_terms(W, x_block):
    y = []
    for i, row in enumerate(W):
        terms = []
        for w_ij, x_j in zip(row, x_block):
            if w_ij != "0":
                terms.append(f"{w_ij} {x_j}")
        y.append(" + ".join(terms))
    return y

# Volle Faltung (symbolisch)
x_total = [f"x_{i}" for i in range(12)]
y_full_terms = []
for n in range(len(x_total) + len(w) - 1):
    terms = []
    for m in range(len(w)):
        if 0 <= n - m < len(x_total):
            terms.append(f"{w[m]} {x_total[n-m]}")
    y_full_terms.append(" + ".join(terms))

# Ausgabe pro Block
for p, x_p in enumerate(blocks):
    print(f"=== Block p = {p} ===")
    # W^(0) * x^(p)
    W0 = block_matrix_terms(w, 0, M)
    y0 = matvec_terms(W0, x_p)
    
    # c_e^(p)
    ce_terms = ["0"] * M
    for k in range(1, p+1):
        Wk = block_matrix_terms(w, k, M)
        x_pk = blocks[p-k]
        yk = matvec_terms(Wk, x_pk)
        ce_terms = [f"({a}) + ({b})" if a!="0" else b for a, b in zip(ce_terms, yk)]

    # Gesamtes e^(p)
    e_p = [f"({y0[i]}) + ({ce_terms[i]})" if ce_terms[i]!="0" else y0[i] for i in range(M)]

    # Volle Faltung Teil
    y_ref = y_full_terms[p*M:(p+1)*M]

    print("\\begin{align}")
    for i in range(M):
        print(f"e^{p}[{i}] &= {e_p[i]} \\\\")
    print("\\end{align}\n")

    print("% Volle Faltung (Referenz)")
    print("\\begin{align}")
    for i, expr in enumerate(y_ref):
        print(f"y[{p*M + i}] &= {expr} \\\\")
    print("\\end{align}\n")