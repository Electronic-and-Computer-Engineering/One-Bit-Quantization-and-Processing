import numpy as np

def sgn0(v):
    return np.where(v >= 0, 1.0, -1.0)