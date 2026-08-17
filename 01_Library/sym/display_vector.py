from IPython.display import display, Math
import sympy as symp

def display_vector(name: str, vec_sym, values=None, show_symbolic=True, show_values=True, round_digits=3):
    """
    Display a vector symbolically and/or with values in LaTeX.
    
    Args:
        name: Name of the vector (used in LaTeX output)
        vec_sym: SymPy Matrix of symbolic vector
        values: Optional list or Matrix of values to show
        show_symbolic: Whether to display the symbolic vector
        show_values: Whether to display evaluated vector
        round_digits: Number of decimal digits to round values
    """
    # Symbolic display
    if show_symbolic:
        display(Math(r"\mathbf{" + name + r"} = " + symp.latex(vec_sym.transpose()) + r"^T"))

    # Value-based display
    if show_values and values is not None:
        # Ensure it's a SymPy Matrix
        if not isinstance(values, symp.Matrix):
            values = symp.Matrix(values)
        values = values.evalf(round_digits)
        display(Math(r"\mathbf{" + name + r"} = " + symp.latex(values.transpose()) + r"^T"))