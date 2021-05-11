# travelling-thief-problem

Project code for Team Tech Lab EXE in the travelling thief competition.

# Experiment running instructions.

First, compile the cython code. 

    $ cythonize -i -3 core.pyx && mv *.so core.so

Then open up three terminals and run your experiments. eg. 

    $ ./experiment.py 1
    $ ./experiment.py 2
    $ ./experiment.py 3

And Kyle will do the same for 4, 5 and 6.

When the run has completed, you can should it again by repeating the same command. We want to make a lot of runs.

If you want to leave it for longer without having to repeat the command, you can add a second parameter to the input which specifies the number of runs to make, eg. 

    $ ./experiment.py 1 100000

The default is 10000, so this will run 10 times as long.


Data will be stored every 1000 algo runs, and also every hour.


# General notes

`core.pyx`/`core_cpython.py` contains the framework.

`visualise` has scripts for generating visualisations, and stores graph output.

`benchmark` contains scripts for benchmarking.

`output` contains raw output files, and some examples for testing.

`tests` has scripts for testing.

`resources` contains information that defines each problem.

`misc` has manually generated calculations for verifying that the code works.

`experiment/experiment.py` is the final experiment script.

`example_generation` contains scripts for generating example solutions, for testing.

`distance_matrices` stores distance matrix calculations, to reduce start-up times.

All data structures are 1-indexed in analogy to the mathematics. We may want to cut them down to 0-indexed for performance, but it'll be confusing. Could we write an abstraction class over Numpy arrays that just shifts the indices? That way we'd get best of both worlds.


# Core code documentation.

### Solution

Holds one possible x- and z-value, and calculates the corresponding fitness function values.
Uses data structures passed from an associated `Problem` object.

`Solution.run` Does the calculation of the fitness functions.

### BigSolution

Drop-in (almost) replacement for `Solution`, optimised for calculation on the largest problems. It uses a lazily-calculated sparse distance matrix rather than a pre-calculated dense one.
Used for 33810-city problems.

### Problem

Holds all of the data structures needed to run the experiments on a particular problem. Reads them from files or calculates them as needed.

Contains a population of `Solution` or `BigSolution` objects (depending on problem size), which are evolved with the `Problem.algo` method.

Also has various IO and auxiliary methods. The logic for generating random initial solutions belongs to this class.

# Notes on optimisations

1. Sparse matrix for bigger problems, to reduce memory overhead and increase speed.

2. Heavily vectorised and optimised code for calculating the weights, distances and fitness functions.

3. Calculation of `a` makes use of fast integer arithmetic. Was previously a dictionary mapping between cities and items, but that was slower.




# Code style guide.

100 character limit on each line. Otherwise, PEP 8 and other Python standards.

Use Black to reformat code, and Flake as a linter. Where they disagree, ignore Flake.

eg. `$ Black .` and `$ flake8 --max-line-length 100 --exclude "tests/*, gecco19-thief/*"`.

# to do list

### General

-- Everyone try compiling and using the `core.pyx` file using the instructions below. Let me know if there are problems. It might work differently on different operating systems.

-- Everyone Move over to Cython.

### Code quality improvements (Q)

Q1. Write tests so that we can make changes/optimisations without worrying about breaking code

Q2. Write better documentation and comments.

Q3. Come up with example solutions for the `a280-n1395` problem and calculate f and g by hand. We need these for testing, so make sure that our calculations are correct.

### Performance optimisation (P)


P1. -- on hold for now, unless someone has ideas -- Improve speed of framework by optimising code where needed. Perhaps multiprocessing? Or we can do more things with matrices or write better code.

P3. Profile `core.py` to find out where time is spent and where optimisations are needed.

P5. Increase speed of `BigSolution.d_calc`.

### Algorithm improvements and research (A)

A1. Try to find improvements in evolutionary algorithm. eg. which tournament style and selection do we want? Using a highly dominance-based approach with Pareto fronts currently, but this is too strong I think. Perhaps binary tournament selection with dominance as the comparison operator (random choice if tied)?

A2. Figure out how crossover should be done on x (permutation). Did the lectures cover this? There are approaches in the literature. For example we can split the permutation into cycles and choose some of these. Someone needs to figure out which way is best.

A4. Improve the algorithm with weakest replacement instead of random replacement.

### Framework extensions (F)

F1. There are two different file formats for output: `results/*` and `target/results/*`. Need to figure out which one to use. There is only a minor difference between these.

F2. Write code to calculate the entropy and information value of solutions and populations. Also build metrics to track population diversity. This will enable us to better improve the algorithm, and give us something to write about. Our population appears to be losing too much diversity with each generation currently.


### Visualisations (V)

V1. Investigate what kinds of visualisations we want.

V2. Make visualisations that show the route of a particular x-value.

V3. Make visualisations that show how the information content changes over time. (after completion of F2).


### Misc (M)

M1. Exploratory data analysis. Try to find patterns in the input data. For example, it would be good to know the average weights and profits of the items in each problem, and the time of a typical route.



# Optimisation


`experiment_time_optimisation.py` is used to find out which parts need optimising. Re-run after making changes and see the difference. To run on a specific problem n, use `./experiment_time_optimisation n`.

`multi_time.xsh` uses `experiment_time_optimisation.py` on each problem in sequence, producing a benchmark for each problem which is saved in `output/run_times.txt`. If Xonsh is installed, it can be run with `./multi_split.xsh`. If you don't know the Xonsh language, don't worry too much about how it works.

`experiment_set_up_time.py` is no longer used, since the calculations that benchmarks are very fast now. Code relating to the `Problem` class has mostly been optimised satisfactorily.

The `Solution` class code needs optimising a lot, which may involve changing the `Problem` code. In particular, `Solution.run` needs to be much faster, and maybe we can run several in parallel.

PyPy failed to increase speed. Cython was successful.


# Move over to Cython, and Python version to use. 

If the group agrees, we will replace `core.py` with `core.pyx` written in Cython. It would be a good idea to use predominantly Python-compliant code for readability, with Cython-extended syntax planned to only used for `Solution.run`, if at all.

For now, the Cython copy is named `core_cython.pyx`. This will be changed to `core.pyx` if and when we agree to move over.

To compile `core.pyx`, use the following command (you will need to install GCC first):

    $ cythonize -i -3 core.pyx && mv *.so core.so

This will generate `core.so`. Then you can simply use `import core` to import this as usual. You might get warnings about your Python version number, and we might want to figure out how to fix this on Mac OS, though it's not a priority.

We have all agreed to use Python 3.8.

Could everyone try compiling the Cython code on their machines?
