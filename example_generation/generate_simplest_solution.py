"""
Generate simple example solutions for testing. Save in output/example.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import Problem, PROBLEMS, Solution


os.chdir("..")

for problem in PROBLEMS:
    p = Problem(problem)
    num_cities, num_items = p.n, p.m
    x = np.arange(0, num_cities + 1).astype(int)
    z = np.zeros(num_items + 1).astype(bool)
    s = p.solution_type(x, z, p)
    s.run()
    p.output(f"output/example/{problem}.x", f"output/example/{problem}.f", [s])
