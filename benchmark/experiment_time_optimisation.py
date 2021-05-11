#!/usr/bin/env python3.7
"""
Benchmarking tool. Usage: `./experiment_time_optimisation.py n`, choosing the nth problem.
Default n=3. Benchmarks the algorithm as a whole, but currently weighted towards Solution.run.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import *
from time import time

os.chdir("..")

problem_number = 4
if len(sys.argv) > 1:
    problem_number = int(sys.argv[1])

f = PROBLEMS[problem_number]

last = time()
times = []
for _ in range(4):
    p = Problem(f)
    p.population = p.gen_population(50)
    p.algo(10)
    times.append(time() - last)
    last = time()


print(np.mean(times), np.std(times) / len(times))
