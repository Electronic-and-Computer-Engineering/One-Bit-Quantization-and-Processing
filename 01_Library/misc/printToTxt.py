def printToTxt(text, file_path, append = False):
    """
    Print LaTeX formatted text to a file.
    
    Args:
    text (str): LaTeX formatted text to be written to the file.
    file_path (str): Path to the file where the text will be written.
    """
    
    mode = 'a' if append else 'w'
    
    with open(file_path, mode) as file:
        file.write(text)