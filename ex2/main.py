import difflib
from typing import List
import numpy as np


# get grade based on how close was the solution to the desired one
def get_grade(line: List[bool], line_rules: List[int], n: int = 10):
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
    grade_sequence = sm.ratio() + 0.000001
    bits_on_difference = abs(sum(line_rules) - sum(segments))
    grade_bits = (n - bits_on_difference) / n
    return grade_sequence + grade_bits


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
    return np.random.choice(a=[False, True], size=(N, N), p=[p, 1 - p])



def main():
    rows, cols = get_rows_cols_from_txt_file("fish.txt")
    grid = init_grid(N=len(cols), p=0.5)
    print(grid)

    lines = [
        [False, True, True, False, False, True, True, False, False, False],
        [True, True, False, False, False, True, True, False, False, False],
        [False, True, False, False, False, True, True, False, False, False],
        [False, False, False, False, False, True, True, False, False, False],
        [False, False, False, False, False, False, False, True, False, False],
        [True, False, True, False, True, False, False, True, False, False],
        [False, False, False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True, True, True]
    ]
    for i, line in enumerate(lines):
        if i + 1 == 7:
            x = 3
        # print(i + 1, get_grade(line, input))


if __name__ == '__main__':
    main()
