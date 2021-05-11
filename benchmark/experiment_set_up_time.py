"""
Benchmarks the time taken to instantiate a Problem object.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import *
from time import time


os.chdir("..")

for f in PROBLEMS:
    init = time()
    p = Problem(f)
    print(f, time() - init)
