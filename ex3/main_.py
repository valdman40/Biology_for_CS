import numpy as np


class Digit(object):
    def __init__(self, matrix: list[list[int]], value: int):
        self.matrix = matrix
        self.value = value


def gggg(n: int, pattern: str):
    pass
    # retval = []
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         sum = 0
    #         for _ in pattern:
    #             if p[i] == p[j]:
    #                 sum += 1
    #             else:
    #                 sum -= 1
    #         retval.append(sum)
    # return retval


# returns txt file as array where each cell represent line
def get_lines_as_list(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


# takes string and return it as int_array splitted by separator
def from_string_to_int_array(string: str, separator: str = ""):
    splitted = list(string)
    if separator != "":
        splitted = string.split(sep=separator)
    int_array = []
    for s in splitted:
        int_array.append(int(s))
    return int_array


# takes string array and return integers matrix out of it
def from_string_array_to_int_matrix(string_array: list[str]):
    int_matrix = []
    for string in string_array:
        int_matrix.append(from_string_to_int_array(string, ""))
    return int_matrix


def get_matrixes(content: list[list[int]]):
    matrix_list: list[list[list[int]]] = []
    current_matrix: list[list[int]] = []
    for line in content:
        if line == []:
            matrix_list.append(current_matrix)
            current_matrix = []
        else:
            current_matrix.append(line)
    if current_matrix != []:
        matrix_list.append(current_matrix)
    return matrix_list


def get_weight_matrix(matrix: list[list[int]]):
    for pattern in matrix:
        x = gggg(len(pattern), pattern)
        y = 3


def calculate_val(weight_col: list[int], new_line: list[int]):
    sum = 0
    for i, j in zip(new_line, weight_col):
        if i is not None:
            sum += i * j
    return sum

def main():
    # content_as_long_list = from_string_array_to_int_matrix(get_lines_as_list("Digits.txt"))
    # digits: list[list[list[int]]] = get_matrixes(content_as_long_list)
    # digits_with_values = []
    # for i, digit in enumerate(digits):
    #     digits_with_values.append((digit, int(i / 10)))
    #
    # for touple in digits_with_values:
    #     digit, value = touple
    #     print("the matrix")
    #     print(np.matrix(digit))
    #     print(f"represent the value of {value}\n")

    # matrix = [
    #     [0, 0, 1, 0, 1, 0],
    #     [1, 1, 1, 1, 0, 0],
    #     [1, 0, 1, 1, 1, 0],
    #     [0, 1, 0, 0, 0, 1],
    #     [0, 1, 1, 0, 0, 0],
    # ]

    b = calculate_val(weight_col=[5, 4, -3], new_line=[1, None, 1])
    # weight_matrix = get_weight_matrix(matrix)
    n = 0


if __name__ == '__main__':
    main()
