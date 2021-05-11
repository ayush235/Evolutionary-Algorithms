"""
Library of code which computes the algorithm, and provides abstractions to access it.
"""
import numpy as np
import random
from itertools import product
import os
from typing import List, Iterator, Tuple
from scipy.sparse import lil_matrix
import json
import gzip
import subprocess

# Let me know if you want to use a different licence.
# Copyright authors 2020 MIT Licence.

# add name here if you add code.
# Authors: Morgan Downing, Kyle Chen


# List of names for the different problems.
PROBLEMS = [
    "test-example-n4",
    "a280-n279",
    "a280-n1395",
    "a280-n2790",
    "fnl4461-n4460",
    "fnl4461-n22300",
    "fnl4461-n44600",
    "pla33810-n33809",
    "pla33810-n169045",
    "pla33810-n338090",
]


def multi_split(s: str) -> List[str]:
    """Splits on space and tab, removes newline.
    :param s: The string to be split.
    :return: The list of strings, each of which is the characters between each tab or space."""
    return " ".join(s.replace("\n", "").split("\t")).split(" ")


def search_input_last(s: str, lines: List[str]) -> str:
    """Searches the input list for the named entry and returns the last
    tab-separated value in the corresponding line.
    :param s: String to search for.
    :param lines: List of strings to search for.
    :return: Last entry in line starting with the search term."""
    for line in lines:
        if line.startswith(s):
            return multi_split(line)[-1]
    raise NotImplementedError


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Euclidean distance between a and b.
    :param x: a 2d point representing a city.
    :param y: a 2d point representing the other city."""
    return np.sqrt(np.sum((x - y) ** 2))


def v_euclid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector Euclidean distance.
    :param x: Array of 2d points.
    :param y: Array of 2d points.
    :return: Array of distances between xs and ys."""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


class Solution:
    """Represents an individual solution to the TTP."""

    overcapacity_factor = 1.5

    def __init__(self, x: np.ndarray, z: np.ndarray, parent) -> None:
        """
        :param x: permutation, defines route
        :param z: which bags to be picked up in each location
        :param parent: Problem which this Solution is for.
        """
        self.x = x  # route
        self.z = z  # items to pick up
        self.parent: Problem = parent  # parent problem
        self.m = self.parent.m
        self.n = self.parent.n

        assert len(self.x) == self.n + 1
        assert len(self.z) == self.m + 1

        # Fitness functions.
        self.f = 0  # time
        self.g = 0  # profit

        # Extra output variables.
        self.w = 0  # final weight
        self.d = 0  # total distance travelled

        self.has_run = False

    def run(self) -> None:
        """Traces the paths in the solution, working out the descriptive values
        for it. Probably could be better optimised."""
        # Avoid repeat runs, which waste compute time.
        if self.has_run:
            return
        # Calculate weights.
        ws = self.calc_ws()
        self.w = ws[-1]  # Save weight for output.
        # Calculate objective functions.
        self.g = np.sum(self.z[1:] * self.parent.profits[1:])
        self.f = np.sum(
            self.parent.d[self.x[1:-1], self.x[2:]] / self.v(ws[1:-1])
        ) + self.parent.d[self.x[-1], self.x[1]] / self.v(ws[-1])
        self.d = (
            np.sum(self.parent.d[self.x[1:-1], self.x[2:]])
            + self.parent.d[self.x[-1], self.x[1]]
        )  # Save distance for output. May want to speed up
        if self.w > self.parent.Q:
            self.f *= self.overcapacity_factor
            self.g /= self.overcapacity_factor
        self.has_run = True

    def calc_ws(self) -> np.ndarray:
        """Traverses the route and calculates the weight at each step.
        :return: The array of weights."""
        ws = np.zeros(self.n + 1)
        indices = np.arange(0, self.m)
        # Only use those indices where items are chosen.
        # Because calculating zeros wastes time.
        indices = indices[self.z[1:]]

        # Calculate the weights at each step in the route.
        if len(indices):  # Are any items chosen?
            # Shorten the item weights array to align with indices.
            item_weights = self.parent.weights[self.z]

            # Perform pre-calculations for `a`.
            indices %= self.n - 1
            indices += 2

            cities = self.x[np.arange(2, len(self.x))]
            cities = cities[:, np.newaxis] == indices  # performs `a`
            # Add the new weights.
            ws[2:] += np.array([np.sum(item_weights[x]) for x in cities])
            # Add the weight from the previous city.
            ws = np.cumsum(ws)
        return ws

    @staticmethod
    def x_mutate(x: np.ndarray) -> np.ndarray:
        """Returns a mutation of x.
        :param x: x to mutate.
        :return: Mutated x."""
        x_ = x.copy()
        inds = [random.randint(2, len(x_) - 1) for _ in range(2)]
        x_[inds[0]], x_[inds[1]] = x[inds[1]], x[inds[0]]
        return x_  # todo check maths is right

    @staticmethod
    def z_mutate(z: np.ndarray) -> np.ndarray:
        """Returns a mutation of z.
        :param z: z to mutate.
        :return: Mutated z."""
        z_ = z.copy()
        ind = random.randint(1, len(z_) - 1)
        z_[ind] = 1 - z_[ind]
        return z_

    def mutate(self):
        """Mutates the solution, creating a new one.
        :return: Solution, which has its parameters mutated from the current one."""
        return type(self)(self.x_mutate(self.x), self.z_mutate(self.z), self.parent)

    @staticmethod
    def z_crossover(z: np.ndarray, other_z: np.ndarray):
        z_ind = random.randint(2, len(z) - 1)
        z_new = np.concatenate([z[:z_ind], other_z[z_ind:]])
        other_z_new = np.concatenate([other_z[:z_ind], z[z_ind:]])
        return z_new, other_z_new

    @staticmethod
    def get_cycle(x: np.ndarray, y: np.ndarray, index: int) -> list:
        """Starts at index, follows algorithm to find a cycle."""
        if x[index] == y[index]:
            return []
        start = x[index]
        cycle = []
        current = None
        while current != start:
            cycle.append(index)
            current = y[index]
            index = np.where(x == current)[0][0]  # slow, maybe do a dictionary
        return cycle

    @classmethod
    def get_cycles(cls, x: np.ndarray, y: np.ndarray) -> Iterator[np.ndarray]:
        """Gets all cycles between x and y."""
        done = set()
        for i in range(len(x)):
            if i in done:
                continue
            new_cycle = cls.get_cycle(x, y, i)
            if new_cycle:
                done = done.union(new_cycle)  # probably slow
            yield np.array(new_cycle)

    @classmethod
    def x_partial_crossover(
        cls, x: np.ndarray, y: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """Performs partial mapping crossover between x and y.
        Written by Kyle."""
        # choose the crossover gen segment
        start_index = random.randint(0, len(x) - 1)  # 0 - 9
        end_index = random.randint(start_index + 1, len(y))  # 1 - 10

        x_temp_gen = x[start_index:end_index]
        y_temp_gen = y[start_index:end_index]

        # exchange segment
        x_temp_child = np.concatenate([x[:start_index], y_temp_gen, x[end_index:]])
        y_temp_child = np.concatenate([y[:start_index], x_temp_gen, y[end_index:]])

        # solve conflict

        # 1. mark orignal mapping     e.g. [ [ele_y1, ele_x1], [ele_y2, ele_x2] ]
        x_y_relations = []
        for i in range(len(x_temp_gen)):
            x_y_relations.append([y_temp_gen[i], x_temp_gen[i]])
        y_x_relations = []
        for i in range(len(y_temp_gen)):
            y_x_relations.append([x_temp_gen[i], y_temp_gen[i]])

        # 2. solve conflict
        child1 = cls.solve_conflict(
            x_temp_child, start_index, end_index, y_temp_gen, x_y_relations
        )
        child2 = cls.solve_conflict(
            y_temp_child, start_index, end_index, x_temp_gen, y_x_relations
        )
        return child1, child2

    @classmethod
    def solve_conflict(cls, temp_child, start_index, end_index, temp_gen, relations):
        """Solves conflicts from partial mapping crossover, to prevent duplicates.
        Written by Kyle."""
        # deal with the first segment
        child = np.zeros(len(temp_child)).astype(int)
        for i, e in enumerate(temp_child[:start_index]):
            stop_flag = 0
            for r_ele in relations:
                if e == r_ele[0]:  # exist conflict
                    child[i] = r_ele[
                        1
                    ]  # replace the conflicting element with the element mapped with temp_child
                    stop_flag = 1
                    break
            if stop_flag == 0:  # no conflict
                child[i] = e  # keep the original position

        # deal with the last segment
        for j, e in enumerate(temp_child[end_index:]):
            stop_flag = 0
            for r_ele in relations:
                if e == r_ele[0]:  # exist conflict
                    child[j + end_index] = r_ele[
                        1
                    ]  # replace the conflicting element with the element mapped with temp_child
                    stop_flag = 1
                    break
            if stop_flag == 0:  # no conflict
                child[j + end_index] = e  # keep the original position

        child = np.concatenate([child[:start_index], temp_gen, child[end_index:]])
        if len(child) > len(np.unique(child)):
            child = cls.solve_conflict(
                child, start_index, end_index, temp_gen, relations
            )
        return child

    @classmethod
    def x_cycle_crossover(
        cls, x: np.ndarray, y: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """Performs cycle crossover between x and y."""
        x, y = x[2:], y[2:]
        x_, y_ = x.copy(), y.copy()
        switch = True
        for cycle in cls.get_cycles(x, y):
            if switch and cycle.size > 0:
                x_[cycle], y_[cycle] = y_[cycle], x_[cycle]
            switch = not switch
        return np.concatenate([np.array([0, 1]), x_]), np.concatenate(
            [np.array([0, 1]), y_]
        )

    def crossover(self, other):
        """Crossover with another solution.
        Check lectures for correct approach.
        Method should be symmetric.
        :return: A Solution which is a combination of the current two."""
        z_1, z_2 = self.z_crossover(self.z, other.z)
        if self.parent.crossover_type == "cycle":
            x_1, x_2 = self.x_cycle_crossover(self.x, other.x)
        elif self.parent.crossover_type == "partial":
            x_1, x_2 = self.x_partial_crossover(self.x, other.x)
        else:
            raise NotImplementedError
        return type(self)(x_1, z_1, self.parent), type(self)(x_2, z_2, self.parent)

    def v(self, w):
        """Velocity calulation.
        :param w: current weight. Float or array.
        :return: v, speed. Float or array, the same shape as w."""
        v = self.parent.v_max - (w / self.parent.Q) * (
            self.parent.v_max - self.parent.v_min
        )
        return np.maximum(v, self.parent.v_min)

    def __str__(self) -> str:
        """Print solution parameters.
        :return: The parameters in string form, as three lines."""
        return (
            f"{' '.join(str(s) for s in self.x[1:])}\n"
            + f"{' '.join(str(s) for s in self.z[1:].astype(int))}\n\n"
        )

    def __repr__(self) -> str:
        """Print solution objective values.
        :return: Space-separated string of function values."""
        return f"{self.f :0<16} {self.g :0<16}\n"

    def __gt__(self, other) -> bool:
        """Calculates dominance.
        :param self: Solution
        :param: other: Solution
        :return: True if self dominates other."""
        return (self.f <= other.f and self.g >= other.g) and (
            self.f < other.f or self.g > other.g
        )


class BigSolution(Solution):
    """Variant of Solution to be used on the biggest problems. Uses a sparse distance matrix, which
    is calculated point by point as needed."""

    def run(self) -> None:
        """Traces the paths in the solution, working out the descriptive values
        for it. Probably could be better optimised."""
        if self.has_run:
            return

        ws = self.calc_ws()
        self.w = ws[-1]

        self.g = np.sum(self.z[1:] * self.parent.profits[1:])
        self.f = (
            np.sum(
                self.d_calc(self.x[1:-1], self.x[2:], self.parent.d, self.parent.points)
                / self.v(ws[1:-1])
            )
            + (
                self.d_calc(
                    np.array([self.x[-1]]),
                    np.array([self.x[1]]),
                    self.parent.d,
                    self.parent.points,
                )
                / self.v(ws[-1])
            )[0]
        )
        self.d = (
            np.sum(
                self.d_calc(self.x[1:-1], self.x[2:], self.parent.d, self.parent.points)
            )
            + self.d_calc(
                np.array([self.x[-1]]),
                np.array([self.x[1]]),
                self.parent.d,
                self.parent.points,
            )[0]
        )
        self.has_run = True
        if self.w > self.parent.Q:
            self.f *= self.overcapacity_factor
            self.g /= self.overcapacity_factor

    @staticmethod
    def d_calc(a: np.ndarray, b: np.ndarray, d: lil_matrix, points: List[np.ndarray]):
        """Lazily calculate distances as needed. This is probably very slow. We can optimise.
        :param a: First axis slice.
        :param b: Second axis slice.
        :param d: Sparse distance matrix.
        :param points: List of points, each an array. Want to convert the whole thing to array?"""
        d_bool = d[a, b].toarray() == 0
        if np.any(d_bool):
            a_ = []
            b_ = []
            a_i = []
            b_i = []
            for i, (a_0, b_0) in enumerate(zip(a, b)):
                if d_bool[0, i]:
                    a_.append(points[a_0])
                    a_i.append(a_0)
                    b_.append(points[b_0])
                    b_i.append(b_0)
            a_ = np.array(a_)
            b_ = np.array(b_)
            a_i = np.array(a_i)
            b_i = np.array(b_i)
            d[a_i, b_i] = v_euclid(a_, b_)
        return d[a, b].toarray()[0, :]


class Problem:
    """Handles IO and problem-wide variables."""

    def __init__(
        self, file: str, big_override: bool = False, crossover_type: str = "cycle"
    ) -> None:
        """
        :param file: String representing filename without extension,
            eg. "a280-n279"
        :param big_override: Used for testing. Forces sparse matrix, even for small problems.
        :param crossover_type: Choose which x crossover method to use. "cycle" or "partial".
        """
        with open(f"resources/{file}.txt", "r") as fin:
            lines = fin.readlines()

        self.name = file

        # Process special named variables at top of file.
        self.Q = int(search_input_last("CAPACITY", lines))
        self.v_min = float(search_input_last("MIN", lines))
        self.v_max = float(search_input_last("MAX", lines))

        # Boolean is_big is True if we should use sparse matrices (for big problems).
        self.is_big = self.name.startswith("pla33810")

        # Special lengths that are important for our representation vectors.
        self.m = int(search_input_last("NUMBER OF ITEMS", lines))
        self.n = int(search_input_last("DIMENSION", lines))

        # Process main bulk of data in file.
        lines = [multi_split(x) for x in lines]
        i1 = next(i for i, x in enumerate(lines) if x[0].startswith("NODE_"))
        i2 = next(i for i, x in enumerate(lines) if x[0].startswith("ITEMS"))
        # node_lines pertain to cities, item_lines to items
        node_lines, items_lines = lines[i1 + 1 : i2], lines[i2 + 1 :]

        points = [np.array([float(y) for y in x[1:]]) for x in node_lines]
        # Calculate distances between cities.
        if not self.is_big and not big_override:
            d_path = f"distance_matrices/{self.name}"
            if os.path.exists(f"{d_path}.npy"):
                self.d = np.load(f"{d_path}.npy")
            else:
                self.d = self._calc_weight_matrix(
                    points
                )  # w[0,j]=w[i,0]=0 to avoid off by 1 errors
                np.save(d_path, self.d)
        else:
            self.d = lil_matrix((len(node_lines) + 1, len(node_lines) + 1))
        self.points = [np.array([0, 0])] + points

        # profit, weights
        self.profits, self.weights = self._item_params(items_lines)

        # Where we store Solution objects that make up the current population.
        self.population = []

        start_num_items = self.Q / np.mean(self.weights[1:]) * 0.8
        # What proportion of z should start as 1?
        self.item_fullness = start_num_items / self.m

        # The number of generations that have been applied.
        self.num_iterations = 0

        # Redirects to BigSolution class for big problems.
        self.solution_type = BigSolution if self.is_big else Solution
        if big_override:
            self.solution_type = BigSolution

        # The crossover method to use for x.
        self.crossover_type = crossover_type

    def gen_x(self) -> np.ndarray:
        """Generates a value for an x parameter.
        :return: The x array."""
        return np.concatenate(
            (np.array([0, 1]), np.random.permutation(np.arange(2, self.n + 1)))
        )

    def gen_z(self) -> np.ndarray:
        """Generates a value for a z parameter.
        Each entry has self.item_fullness probability of being 1.
        :return: The z array."""
        return np.concatenate((np.array([0]), np.random.random(self.m))) > (
            1 - self.item_fullness
        )

    def gen_solution(self) -> Solution:
        """Generates a Solution.
        :return: A random Solution or BigSolution."""
        return self.solution_type(self.gen_x(), self.gen_z(), self)

    def gen_population(self, n: int) -> List[Solution]:
        """Generates n random solutions.
        :param n: the number of Solutions to generate.
        :return: A list of Solutions."""
        return [self.gen_solution() for _ in range(n)]

    def pareto(self) -> List[Solution]:
        """Calculates the Pareto front.
        :return: A list of Solutions which form the Pareto front."""
        return [s for s in self.population if not any(o > s for o in self.population)]

    def output(self, paramfile: str, outfile: str, selected: List[Solution]) -> None:
        """Write selected solutions two two output files, one of parameters and
        the other of objective values.
        :param paramfile: File to output parameters.
        :param outfile: File to output objective function values.
        :param selected: Solutions to be output."""
        if not selected:
            selected = self.population
        selected = sorted(selected, key=lambda x: x.f)
        for s in selected:
            s.run()
        with open(paramfile, "w") as param:
            with open(outfile, "w") as out:
                for s in selected:
                    param.write(str(s))
                    out.write(repr(s))

    def algo(self, num_iterations: int) -> None:
        """Evolutionary algorithm implementation.
        Can be swapped out if a better one is found."""
        # Calculate objective functions for the start population.
        self.run_population()
        for iteration in range(num_iterations):
            # Basic selection method, using each from Pareto combined with one from general pop.
            pareto = self.pareto()
            for s in pareto:
                other = random.choice(self.population)

                # Mutation.
                new_s = s.mutate()

                # Crossover.
                new_s = new_s.crossover(other)[0]  # Can change to 2 offspring.
                new_s.run()

                # Custom dominance-based weak replacement, suggested by Shoma.
                non_pareto = [x for x in self.population if x not in pareto]
                if non_pareto:
                    pivot = random.choice(non_pareto)
                    dominated = [x for x in self.population if pivot > x]
                    if dominated:
                        other2 = random.choice(dominated)
                    else:
                        other2 = pivot
                else:
                    other2 = random.choice(pareto)
                self.population[self.population.index(other2)] = new_s
        self.num_iterations += num_iterations

    def read_sols_from_file(self, filename: str) -> None:
        """Reads example solution parameters from a file.
        Note: not implemented for big problems.
        :param filename: File to read."""

        def string_to_arr(s: str) -> np.ndarray:
            """Converts a string into a parameter array.
            :param s: A string consisting of a space-separated sequence of integers.
            :return: An array of the form needed for x or z."""
            return np.array([0] + [int(x) for x in s.split(" ")])

        with open(filename, "r") as fin:
            lines = fin.readlines()
        xs, zs = lines[::3], lines[1::3]
        self.population = [
            self.solution_type(string_to_arr(xs[i]), string_to_arr(zs[i]) > 0.5, self)
            for i in range(len(xs))
        ]
        self.run_population()

    def run_population(self) -> None:
        """Call the run method on every Solution within the population."""
        for x in self.population:
            x.run()

    def reinit(self) -> None:
        """Return the Problem object to its initial configuration after runs.
        Much more efficient than creating a new object."""
        self.population = []

    @staticmethod
    def _z_to_list(z):
        """
        Converts z to a compact list representation, so that it can be stored in JSON.
        :param z: z parameter for the Solution.
        :return: The indices where z is True.
        """
        return [i for i in range(len(z)) if z[i]]

    @staticmethod
    def _z_from_list(z_l, m):
        """
        Converts z from a list back to an array.
        :param z_l: The list representation from Problem._z_to_list.
        :param m: The number of items.
        :return: z.
        """
        z = np.zeros(m + 1).astype(bool)
        z[np.array(z_l).astype(int)] = True
        return z

    def to_json(self) -> str:
        """Dumps the population into JSON format.
        :param compression: Determines whether the output should be compressed.
        :return: The JSON as a string."""
        self.run_population()
        pop = [
            {
                "x": s.x.tolist(),
                "z": self._z_to_list(s.z),
                "f": float(s.f),
                "g": float(s.g),
                "w": float(s.w),
                "d": float(s.d),
            }
            for s in self.population
        ]
        return json.dumps((self.num_iterations, pop))

    def from_json(self, s_str: str) -> (int, List[Solution]):
        """Reads a population from JSON format into self.population.
        :param s_str: The JSON to read.
        :param compression: Determines whether the input is compressed and should be expanded.
        :return: the iteration number and the list of Solutions."""
        num_iterations, pop = json.loads(s_str)
        solutions = []
        for s_dict in pop:
            s = self.solution_type(
                np.array(s_dict["x"]), self._z_from_list(s_dict["z"], self.m), self
            )
            s.f = s_dict["f"]
            s.g = s_dict["g"]
            s.w = s_dict["w"]
            s.d = s_dict["d"]
            s.has_run = True
            solutions.append(s)
        return num_iterations, solutions

    def iterations_from_json(
        self, filename: str
    ) -> Iterator[Tuple[int, List[Solution]]]:
        """
        Reads all saved iterations from a JSON file.
        :param filename: The JSON file to read.
        :return: A generator consisting of the iteration number, and a list of the Solutions at
        each iteration.
        """
        with open(filename, "r") as fin:
            for line in fin:
                yield self.from_json(line.strip("\n"))

    def last_iteration_from_json(self, filename: str) -> (int, List[Solution]):
        """
        Reads the final iteration from a JSON file.
        :param filename: The JSON file to read.
        :return: The iteration number and list of Solutions from the final iteration.
        """
        with subprocess.Popen(
            "/bin/sh", stdin=subprocess.PIPE, stdout=subprocess.PIPE
        ) as proc:
            out = proc.communicate(f"tail -n 1 {filename}".encode("utf8"))
            line = out[0].decode("utf8")
        return self.from_json(line)

    @staticmethod
    def _calc_weight_matrix(points: List[np.ndarray]) -> np.ndarray:
        """Calculates the matrix of distances between each point.
        :param points: Each point is a list of string-floats, which are the coordinates.
        :return: The distance matrix."""
        mat = np.zeros((len(points) + 1, len(points) + 1))
        for i, j in product(
            range(len(points)), range(len(points))
        ):  # calculate each j despite symmetry. Can halve computation later if efficiency needed.
            mat[i + 1, j + 1] = euclidean(points[i], points[j])
        return mat

    @staticmethod
    def _item_params(params: List[List[str]]) -> (np.ndarray, np.ndarray):
        """Read item parameters from string and load into data structures.
        :param params: A list for each item, with information concerning the item and related city.
        :return: Tuple of weight and profit arrays, and the mapping of cities to items."""
        weights, profits = np.zeros(len(params) + 1), np.zeros(len(params) + 1)
        for i, line in enumerate(params):
            weights[i + 1] = int(line[2])
            profits[i + 1] = int(line[1])
        return profits, weights
