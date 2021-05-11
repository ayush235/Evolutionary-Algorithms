"""
Calculate f and g given x and z.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import Problem, PROBLEMS, Solution


os.chdir("..")


problem = PROBLEMS[0]
x = np.array([0, 1, 2, 3, 4])
z = np.array([False, True, False, False])


p = Problem(problem)
s = p.solution_type(x, z, p)
s.run()

print(s.f)
print(s.g)
print(s.w)
