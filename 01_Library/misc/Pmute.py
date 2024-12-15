# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:42:29 2024

@author: mayerflo
"""

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__