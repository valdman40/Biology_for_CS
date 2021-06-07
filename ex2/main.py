import difflib
from typing import List
from matplotlib.colors import ListedColormap
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


class Grid(object):
    def __init__(self, g: list[list[bool]], n, fitness=0):
        self.grid = g
        self.N = n
        self.fitness = fitness

    def get_grid(self):
        return self.grid

    def get_grade(self):
        return self.fitness

    def get_fitness_line_By_line(self, lines: list[list[bool]], rules: list[list[int]]):
        fitness = 0
        for i, (line, rule) in enumerate(zip(lines, rules)):
            fitness += self.get_line_fitness(line, rule)
        return fitness

    def set_fitness(self, rows_rules: list[list[int]], cols_rules: list[list[int]]):
        grid_rows_fitness = self.get_fitness_line_By_line(self.grid, rows_rules)
        grid_by_cols = zip(*self.grid)
        grid_cols_fitness = self.get_fitness_line_By_line(grid_by_cols, cols_rules)
        grid_fitness = grid_rows_fitness + grid_cols_fitness
        self.fitness = grid_fitness

    def get_fitness(self):
        return self.fitness

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
        retval = get_fitness_bits_difference(segments, line_rules, self.N) + get_list_similarity(segments, line_rules) * 100
        return retval

    def improve_line(self, line: list[bool], rule: list[int]):
        # switch 1 bit
        if get_lists_sum_difference(line, rule) > 0:
            i = random.randrange(0, self.N)
            line[i] = not line[i]
        # replace 2 bits
        if get_list_similarity(line, rule) < 1:
            i, j = random.randrange(0, self.N), random.randrange(0, self.N)
            line[i], line[j] = line[j], line[i]

    def improve(self, line_rules: list[list[int]]):
        for line, rule in zip(self.grid, line_rules):
            self.improve_line(line, rule)

    def copy(self):
        copied = []
        for g in self.grid:
            copied.append(g.copy())
        return Grid(copied, self.N, self.fitness)

    def mutate(self, p):
        for line in self.grid:
            for i in range(self.N):
                line[i] = random.choices([line[i], not line[i]], [p, 1 - p], k=1)[0]


# returns similarity between 2 lists
def get_list_similarity(segments: List[int], line_rules: List[int]):
    sm = difflib.SequenceMatcher(None, segments, line_rules)
    grade_sequence = sm.ratio()
    return grade_sequence


def get_sum_different(l1: list[int], l2: list[int]):
    return abs(sum(l1) - sum(l2))


def get_lists_sum_difference(segments: List[int], line_rules: List[int]):
    bits_on_difference = get_sum_different(line_rules, segments)
    return bits_on_difference


# returns fitness by bits similarity
def get_fitness_bits_difference(segments: List[int], line_rules: List[int], n: int = 10):
    bits_on_difference = get_lists_sum_difference(segments, line_rules)
    grade_bits = (n - bits_on_difference) / n
    return grade_bits


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
def cross_over(parent1: Grid, parent2: Grid, N: int):
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
        new_grids.append(cross_over(grid_parent1, grid_parent2, N))
    return new_grids


def calculate_grade_for_each_grid(grids: list[Grid], rows_rules: list[list[int]], cols_rules: list[list[int]]):
    best_grid: Grid = grids[0]  # init
    for grid in grids:
        grid.set_fitness(rows_rules, cols_rules)  # so each grid will have grade calculated
        if grid.get_grade() > best_grid.get_grade():
            best_grid = grid
    return best_grid


current_gen_girds = []
best_grid_all_gen = None
population_size = 100
total_frames = 0


# activates each life cycle
def life_cycle(frameNum, img, N: int, rows_rules: list[list[int]], cols_rules: list[list[int]]):
    global current_gen_girds, best_grid_all_gen, total_frames
    total_frames += 1

    # improve each grid
    for grid in current_gen_girds:
        grid.improve(line_rules=rows_rules)

    # get best grid
    best_grid_current_gen = calculate_grade_for_each_grid(current_gen_girds, rows_rules, cols_rules)
    if best_grid_current_gen.get_grade() > best_grid_all_gen.get_grade():
        best_grid_all_gen = best_grid_current_gen.copy()
        print('best %.2f' % best_grid_all_gen.get_grade(), "| frame number %4d" % total_frames)

    if total_frames % 200 == 0:
        print("--------frame number %4d--------" % total_frames)

    current_gen_girds = prepare_next_generation(current_gen_girds, N)
    img.set_data(best_grid_all_gen.get_grid())
    return img,


def main():
    global current_gen_girds, best_grid_all_gen
    rows, cols = get_rows_cols_from_txt_file("5x5_1.txt")
    N = len(cols)  # since every1 is square
    for _ in range(population_size):
        current_gen_girds.append(Grid(init_grid(N=N, p=0.5), N))
    best_grid_all_gen = current_gen_girds[0]  # init best grid
    figure, axes = plt.subplots()
    cmap = ListedColormap(['w', 'k'])
    img = axes.imshow(current_gen_girds[0].get_grid(), interpolation='nearest', cmap=cmap)
    ani = animation.FuncAnimation(figure, life_cycle, fargs=(img, N, rows, cols),
                                  frames=10,
                                  interval=10,  # millisecond to interval
                                  save_count=50,
                                  repeat=True)
    plt.show()


if __name__ == '__main__':
    main()
