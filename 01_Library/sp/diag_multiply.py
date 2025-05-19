import numpy as np

def diag_multiply(arg1: np.ndarray, arg2: np.ndarray, mode: str = "left") -> np.ndarray:
    """
    Performs stable diagonal matrix multiplication depending on mode.
    
    Parameters:
    - arg1 : np.ndarray
        - If mode == "left": vector of shape (N,)
        - If mode == "right": matrix of shape (N, M)
    - arg2 : np.ndarray
        - If mode == "left": matrix of shape (N, M)
        - If mode == "right": vector of shape (M,)
    - mode : str
        Either "left" (diag(vector) @ matrix) or "right" (matrix @ diag(vector))

    Returns:
    - result : np.ndarray
        The result of the diagonal matrix multiplication.
    """
    mode = mode.lower()

    if mode == "left":
        vector = np.asarray(arg1)
        matrix = np.asarray(arg2)
        if vector.ndim != 1:
            raise ValueError("For 'left' mode, the first argument must be a 1D vector.")
        if vector.shape[0] != matrix.shape[0]:
            raise ValueError(f"Vector length {vector.shape[0]} must match number of matrix rows {matrix.shape[0]}.")
        return vector[:, None] * matrix  # row-wise scaling

    elif mode == "right":
        matrix = np.asarray(arg1)
        vector = np.asarray(arg2)
        if vector.ndim != 1:
            raise ValueError("For 'right' mode, the second argument must be a 1D vector.")
        if vector.shape[0] != matrix.shape[1]:
            raise ValueError(f"Vector length {vector.shape[0]} must match number of matrix columns {matrix.shape[1]}.")
        return matrix * vector  # column-wise scaling

    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'left' or 'right'.")