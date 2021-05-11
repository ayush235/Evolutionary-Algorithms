#!/usr/bin/env python3
"""
Transforms results from JSON format into the text format required by the competition.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import Problem, PROBLEMS


os.chdir("..")


for f in PROBLEMS:
    json = f"output/{f}.json"
    if os.path.exists(json):
        p = Problem(f)
        p.population = p.last_iteration_from_json(json)[1]
        p.output(f"output/submission/{f}.x", f"output/submission/{f}.f", p.pareto())
