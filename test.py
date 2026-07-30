import numpy as np

vW = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # len = 8

print(vW[::(len(vW)//2)])  # Stride = 4  →  [1, 5]
print(vW[:len(vW)//2])     # Slice        →  [1, 2, 3, 4]