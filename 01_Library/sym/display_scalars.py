from IPython.display import display, Math
import sympy as symp

def display_scalars(dict_scalars, show_values=True):
    """
    Display scalar variable definitions in LaTeX from a dictionary.
    """
    items = []
    for sym, val in dict_scalars.items():
        if show_values:
            items.append(symp.latex(symp.Eq(sym, val)))
        else:
            items.append(symp.latex(sym))
    latex_string = r"\quad ".join(items)
    display(Math(latex_string))