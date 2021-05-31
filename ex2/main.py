import difflib
from typing import List
from matplotlib.colors import ListedColormap
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_sum_different(l1: list[int], l2: list[int]):
    return abs(sum(l1) - sum(l2))


def get_bits_difference(line: List[bool], line_rules: List[int]):
    segments = []
    segment_length = 0
    try:
        for square in line:
            if not square:  # if we're at 0
                segments.append(segment_length)
                segment_length = 0
            else:
                segment_length += 1
        if segment_length > 0:
            segments.append(segment_length)
    except:
        x =3

    bits_on_difference = get_sum_different(line_rules, segments)
    return bits_on_difference


def get_grade_bits(line: List[bool], line_rules: List[int], n: int = 10):
    bits_on_difference = get_bits_difference(line, line_rules)
    grade_bits = (n - bits_on_difference) / n
    return grade_bits


def get_grade_sequence(line: List[bool], line_rules: List[int], n: int = 10):
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
    sm = difflib.SequenceMatcher(None, segments, line_rules)
    grade_sequence = sm.ratio()
    return grade_sequence


# get grade based on how close was the solution to the desired one
def get_grade(line: List[bool], line_rules: List[int], n: int = 10):
    retval = get_grade_bits(line, line_rules, n) + get_grade_sequence(line, line_rules, n)
    return retval


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


def get_grades(grid: list[list[bool]], rules: list[list[int]], N: int):
    grade = dict()
    for i, (line, rule) in enumerate(zip(grid, rules)):
        grade[f"line_{i}"] = (line, get_grade(line, rule, n=N))
    return grade


best_probability = 0  # for myself to see if i got better


# returns array of lines according to their probablity
def display_each_value_by_probability(rows_grade: dict[str, (list[bool], int)]):
    global best_probability  # for myself to see if i got better
    arr = []
    probabilities = dict()
    total_sum = sum([x[1] for x in rows_grade.values()])
    for key in rows_grade.keys():
        original_line = rows_grade[key][0]
        probability = (rows_grade[key][1] / total_sum) * 100
        # for myself to see if i got better
        if probability > best_probability:
            best_probability = probability
            print('best', best_probability)
        probabilities[key] = (original_line, probability)
    for val in probabilities.values():
        line, probability = val
        for i in range(int(probability)):
            arr.insert(0, line)
    return arr


# generates child from parents details
def get_new_child(parent1: list[bool], parent2: list[bool]):
    place_to_cut = random.randrange(0, len(parent1))
    new_child = parent1[:place_to_cut] + parent2[place_to_cut:]
    return new_child


# return new grid after changing it
def get_new_grid_by_grades(rows_grade: dict[str, (list[bool], int)], N: int):
    new_grid = []
    arr = display_each_value_by_probability(rows_grade)
    for i in range(N):
        parent1 = random.choice(arr)
        parent2 = random.choice(arr)
        new_grid.append(get_new_child(parent1, parent2))
    return new_grid


def try_to_improve(grid: list[list[bool]], line_rules: list[list[int]]):
    improved_grid = []
    for line, rule in zip(grid, line_rules):
        length = len(line)
        if get_bits_difference(line, rule) > 0:
            i = random.randrange(0, length)
            line[i] = not line[i]
        elif get_grade_sequence(line, rule, length) < 1:
            i = random.randrange(0, length)
            j = random.randrange(0, length)
            line[i], line[j] = line[j], line[i]
        improved_grid.append(line)
    return improved_grid


def update(frameNum, img, grid: list[list[bool]], N: int, rows_rules: list[list[int]], cols_rules: list[list[int]]):
    improved_grid = grid
    for _ in range(50):
        improved_grid = try_to_improve(improved_grid, rows_rules)
    rows_grade = get_grades(improved_grid, rows_rules, N)
    new_grid = get_new_grid_by_grades(rows_grade, N)
    # cols_grade = get_grades(zip(*grid), rows_rules, N)

    img.set_data(new_grid)
    return img,


def main():
    rows, cols = get_rows_cols_from_txt_file("fish.txt")
    grid = init_grid(N=len(cols), p=0.5)
    figure, axes = plt.subplots()
    cmap = ListedColormap(['w', 'k'])
    img = axes.imshow(grid, interpolation='nearest', cmap=cmap)
    ani = animation.FuncAnimation(figure, update, fargs=(img, grid, len(grid), rows, cols),
                                  frames=10,
                                  interval=10,  # millisecond to interval
                                  save_count=50,
                                  repeat=True)
    plt.show()


if __name__ == '__main__':
    main()
