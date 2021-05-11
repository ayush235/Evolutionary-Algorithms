#!/usr/bin/env python3
"""
Final run experimental run of algorithm. Saves output in JSON format.
Usage `./experiment.py n`, for experiment n. eg. `./experiment.py 1`.
Optional parameter number of runs eg. `./experiment.py 1 100000` to run the experiment 100000 times.
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core_cpython import Problem, PROBLEMS


os.chdir("..")


ITERATION_FACTOR = 10
SAVE_FACTOR = 1000


def write(prob):
    with open(OUT_FILE, "a") as fin:
        fin.write(f"{prob.to_json()}\n")


# Read input.
if len(sys.argv) > 1:
    f = PROBLEMS[int(sys.argv[1])]
else:
    sys.exit(1)
if len(sys.argv) > 2:
    num_runs = int(sys.argv[2])
    num_loops = int(math.floor(num_runs / ITERATION_FACTOR))
else:
    num_loops = 1000

OUT_FILE = f"output/{f}.json"


# Initialise the Problem object. Continue from previous run if available.
p = Problem(f)
if os.path.exists(OUT_FILE):
    start_bool = False
    with open(OUT_FILE, "r") as fin:
        line = fin.read().splitlines(keepends=False)[-1]
    p.num_iterations, p.population = p.from_json(line)
else:
    start_bool = True
    p.population = p.gen_population(100)


# Run experiment
if start_bool:
    write(p)
last = time.time()
for i in range(num_loops):
    print(i)
    p.algo(ITERATION_FACTOR)
    if i != num_loops and p.num_iterations % SAVE_FACTOR == 0:
        write(p)
        last = time.time()

write(p)
last = time.time()
