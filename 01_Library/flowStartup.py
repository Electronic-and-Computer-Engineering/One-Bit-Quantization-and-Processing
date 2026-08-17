# IPython startup script
# Automatisches Laden benutzerdefinierter Tools

import sys
import os

# === Benutzerdefinierter Pfad zur Library ===
my_lib_path = os.path.expanduser("C:/Users/mayerflo/OneDrive - FH JOANNEUM/FHJ/DISS_FM/3_CodeARepo/01_Library/globalTools/")
if my_lib_path not in sys.path:
    sys.path.insert(0, my_lib_path)

# === Automatischer Import ===
try:
    from plottools import *
    print("✔ plottools loaded.")
except ImportError:
    print("plottools not found. CHECK Path")
    
try:
    from mailtools import *
    print("✔ mailtools loaded.")
except ImportError:
    print("mailtools not found. CHECK Path")

try:
    from vecMatcalc import *
    print("✔ vecMatcalc loaded.")
except ImportError:
    print("vecMatcalc not found. CHECK Path")
    
    