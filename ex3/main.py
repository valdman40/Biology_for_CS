import numpy as np


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


def main():
    content_as_long_list = from_string_array_to_int_matrix(get_lines_as_list("Digits.txt"))
    digits: list[list[list[int]]] = get_matrixes(content_as_long_list)
    digits_with_values = []
    for i, digit in enumerate(digits):
        digits_with_values.append((digit, int(i / 10)))

    for touple in digits_with_values:
        digit, value = touple
        print("the matrix")
        print(np.matrix(digit))
        print(f"represent the value of {value}\n")

    x = 3


if __name__ == '__main__':
    main()
