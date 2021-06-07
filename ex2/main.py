import difflib
from typing import List
from matplotlib.colors import ListedColormap
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


def get_fitness_sequence(segments: List[int], line_rules: List[int], n: int = 10):
    sm = difflib.SequenceMatcher(None, segments, line_rules)
    grade_sequence = sm.ratio()
    return grade_sequence


class Grid(object):
    def __init__(self, g: list[list[bool]], n, grade_given=0):
        self.grid = g
        self.N = n
        self.grade = grade_given

    def get_grid(self):
        return self.grid

    def get_grade(self):
        return self.grade

    def get_fitness_line_By_line(self, lines: list[list[bool]], rules: list[list[int]]):
        fitness = 0
        for i, (line, rule) in enumerate(zip(lines, rules)):
            fitness += self.get_line_fitness(line, rule)
        return fitness

    def fitness(self, rows_rules: list[list[int]], cols_rules: list[list[int]]):
        grid_rows_fitness = self.get_fitness_line_By_line(self.grid, rows_rules)
        grid_cols_fitness = self.get_fitness_line_By_line(zip(*self.grid), cols_rules)
        grid_fitness = grid_rows_fitness + grid_cols_fitness
        self.grade = grid_fitness
        return grid_fitness

    # get grade based on how close was the solution to the desired one
    def get_line_fitness(self, line: List[bool], line_rules: List[int]):
        segments = []
        segment_length = 0
        for square in line:
            if not square:  # if we're at 0
                segments.append(segment_length)
                segment_length = 0
            else:
                segment_length += 1
        if segment_length > 0:
            segments.append(segment_length)
        retval = self.get_fitness_bits(segments, line_rules, self.N) + get_fitness_sequence(segments, line_rules,
                                                                                                 self.N) * 100
        return retval

    def get_bits_difference(self, segments: List[int], line_rules: List[int]):
        bits_on_difference = self.get_sum_different(line_rules, segments)
        return bits_on_difference

    def get_fitness_bits(self, segments: List[int], line_rules: List[int], n: int = 10):
        bits_on_difference = self.get_bits_difference(segments, line_rules)
        grade_bits = (n - bits_on_difference) / n
        return grade_bits

    def get_sum_different(self, l1: list[int], l2: list[int]):
        return abs(sum(l1) - sum(l2))

    def improve_line(self, line: list[bool], rule: list[int]):
        # switch 1 bit
        if self.get_bits_difference(line, rule) > 0:
            i = random.randrange(0, self.N)
            line[i] = not line[i]
        # replace 2 bits
        if get_fitness_sequence(line, rule, self.N) < 1:
            i, j = random.randrange(0, self.N), random.randrange(0, self.N)
            line[i], line[j] = line[j], line[i]

    def improve(self, line_rules: list[list[int]]):
        for line, rule in zip(self.grid, line_rules):
            self.improve_line(line, rule)

    def copy(self):
        copied = []
        for g in self.grid:
            copied.append(g.copy())
        return Grid(copied, self.N, self.grade)

    def mutate(self, p):
        for line in self.grid:
            for i in range(self.N):
                line[i] = random.choices([line[i], not line[i]], [p, 1 - p], k=1)[0]


# returns txt file as array where each cell represent line
def get_lines_as_list(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


# takes string and return it as int_array splitted by separator
def from_string_to_int_array(string: str, separator: str):
    splitted = string.split(sep=separator)
    int_array = []
    for s in splitted:
        int_array.append(int(s))
    return int_array


# takes string array and return integers matrix out of it
def from_string_array_to_int_matrix(string_array: list[str]):
    int_matrix = []
    for string in string_array:
        int_matrix.append(from_string_to_int_array(string, " "))
    return int_matrix


# assuming the number is even (means we can split it for half)
# also assume the file is legal and that the matrix is squared (len(rows) = len(cols))
def get_rows_cols_from_txt_file(filename: str):
    content = from_string_array_to_int_matrix(get_lines_as_list(filename))
    cols = content[:len(content) // 2]
    rows = content[len(content) // 2:]
    return rows, cols


def init_grid(N: int, p: float):
    return np.random.choice(a=[False, True], size=(N, N), p=[p, 1 - p]).tolist()


# returns array of lines according to their probablity
def display_each_value_by_probability(grids: list[Grid]):
    arr = []
    probabilities = dict()
    # get total sum of all grades
    total_sum = 0
    for grid in grids:
        total_sum += grid.get_grade()

    for grid in grids:
        probability = (grid.get_grade() / total_sum) * 100
        probabilities[grid] = int(math.ceil(probability))  # rounded
    for grid, probability in probabilities.items():
        for i in range(probability):
            arr.insert(0, grid)
    return arr


# generates child from parents details
def get_new_grid_child(parent1: Grid, parent2: Grid, N: int):
    place_to_cut = random.randrange(0, N)
    parent1_grid = parent1.get_grid()
    parent2_grid = parent2.get_grid()
    new_child = Grid(parent1_grid[:place_to_cut] + parent2_grid[place_to_cut:], N)
    new_child.mutate(p=0.05)
    return new_child


# return new grid after changing it
def prepare_next_generation(grids: list[Grid], N: int):
    new_grids = []
    arr = display_each_value_by_probability(grids)
    for _ in range(len(grids)):
        grid_parent1: Grid = random.choice(arr)
        grid_parent2: Grid = random.choice(arr)
        new_grids.append(get_new_grid_child(grid_parent1, grid_parent2, N))
    return new_grids


def calculate_grade_for_each_grid(grids: list[Grid], rows_rules: list[list[int]], cols_rules: list[list[int]]):
    best_grid: Grid = grids[0]  # init
    for grid in grids:
        grid.fitness(rows_rules, cols_rules)  # so each grid will have grade calculated
        if grid.get_grade() > best_grid.get_grade():
            best_grid = grid
    return best_grid


global_girds = []
best_grid_all_gen = None
number_of_grids = 100
total_frames = 0


def update(frameNum, img, N: int, rows_rules: list[list[int]], cols_rules: list[list[int]]):
    global global_girds, best_grid_all_gen, total_frames
    total_frames += 1

    # improve each grid
    for grid in global_girds:
        grid.improve(line_rules=rows_rules)

    # get best grid
    best_grid_current_gen = calculate_grade_for_each_grid(global_girds, rows_rules, cols_rules)
    if best_grid_current_gen.get_grade() > best_grid_all_gen.get_grade():
        best_grid_all_gen = best_grid_current_gen.copy()
        print('best %.2f' % best_grid_all_gen.get_grade(), "| frame number %4d" % total_frames)

    if total_frames % 200 == 0:
        print("--------frame number %4d--------" % total_frames)

    global_girds = prepare_next_generation(global_girds, N)
    img.set_data(best_grid_all_gen.get_grid())
    return img,


def main():
    global global_girds, best_grid_all_gen
    rows, cols = get_rows_cols_from_txt_file("5x5_1.txt")
    N = len(cols)  # since every1 is square
    for _ in range(number_of_grids):
        global_girds.append(Grid(init_grid(N=N, p=0.5), N))
    best_grid_all_gen = global_girds[0]  # init best grid
    figure, axes = plt.subplots()
    cmap = ListedColormap(['w', 'k'])
    img = axes.imshow(global_girds[0].get_grid(), interpolation='nearest', cmap=cmap)
    ani = animation.FuncAnimation(figure, update, fargs=(img, N, rows, cols),
                                  frames=10,
                                  interval=10,  # millisecond to interval
                                  save_count=50,
                                  repeat=True)
    plt.show()


if __name__ == '__main__':
    main()
