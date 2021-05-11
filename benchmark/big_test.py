"""
Attempt a small run of the algorithm on a big problem, to probe feasibility.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import *
from time import time


os.chdir("..")

init = time()
f = PROBLEMS[7]
p = Problem(f)
p.population = p.gen_population(5)
print("Set up experiment.")
p.run_population()
print(f"initial f average: {np.mean([x.f for x in p.population])}")

p.algo(10)

print("10th generation done.")
print(time() - init)
print(f"Final f average: {np.mean([x.f for x in p.population])}")
