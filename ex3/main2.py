import random
from typing import List

import numpy as np


def get_matrix(filename, num_of_letters):
    with open(filename) as f:
        content = f.readlines()

    number = ""
    all_numbers = []
    i = 0
    for line in content:
        line = line.rstrip("\n")
        number = number + line
        if line == '':
            if i < num_of_letters:
                all_numbers.append(number)
                line_size = len(number)
                number = ""
                i += 1

    if number != "":
        if i < num_of_letters:
            line_size = len(number)
            all_numbers.append(number)

    column_size = len(all_numbers)
    arr = np.zeros(shape=(column_size, line_size), dtype=int)
    for i in range(len(all_numbers)):
        for j in range(len(all_numbers[0])):
            arr[i][j] = all_numbers[i][j]
    return arr


def get_similar(matrix_of_numbers):
    column_size = matrix_of_numbers.shape[1]
    similar_matrix = arr = np.zeros(shape=(column_size, column_size), dtype=int)
    for j in range(matrix_of_numbers.shape[1]):
        curr_check_column = matrix_of_numbers[:, j]
        for k in range(matrix_of_numbers.shape[1]):
            if j != k:
                second_column = matrix_of_numbers[:, k]
                sum_of_similar = 0
                for i in range(len(curr_check_column)):
                    if curr_check_column[i] == second_column[i]:
                        sum_of_similar += 1
                sum_of_different = len(curr_check_column) - sum_of_similar
                grade = sum_of_similar - sum_of_different
            else:
                grade = matrix_of_numbers.shape[1] + 1
            similar_matrix[j][k] = grade
    return similar_matrix


prev_example: List[int] = []
current_example: List[int] = []


def solution(similar_matrix, example, limit_num):
    sum1 = 0
    list_of_numbers: List[int] = []
    for i in range(len(example)):
        list_of_numbers.append(i)

    copy_example = example.copy()
    while len(list_of_numbers) > 0:
        place: int = random.choice(list_of_numbers)
        list_of_numbers.remove(place)
        matrix_column = similar_matrix[:, place]
        for k in range(len(matrix_column)):
            if matrix_column[k] != limit_num:
                x = matrix_column[k] * int(copy_example[k])
                sum1 += matrix_column[k] * int(copy_example[k])
        if sum1 >= 0:
            grade = 1
        else:
            grade = 0
        copy_example[place] = grade
        sum1 = 0

    return copy_example


def change_tenth(example):
    copy_example = example.copy()
    list_of_numbers: List[int] = []
    for i in range(len(copy_example)):
        list_of_numbers.append(i)

    for _ in range(int(len(copy_example) / 10)):
        rand = random.choice(list_of_numbers)
        if copy_example[rand] == '0':
            copy_example[rand] = '1'
        else:
            copy_example[rand] = '0'

    return copy_example


def main():
    global current_example, prev_example
    filename = 'Digits.txt'
    num_of_letters = 3
    matrix_of_numbers = get_matrix(filename, num_of_letters)
    similar_matrix = get_similar(matrix_of_numbers)
    limit_num = similar_matrix.shape[1] + 1

    example = '0000000000011111111001110001100111000010011000011001100001100110000010011110011000011111000000000000'
    list_example = []
    for i in range(len(example)):
        list_example.append(example[i])

    counter = 0
    len_of_example = len(example)
    for j in range(100):
        sum1 = 0
        prev_example = change_tenth(list_example)
        current_example = solution(similar_matrix, prev_example, limit_num)
        i = 0
        while current_example != prev_example:
            prev_example = current_example
            current_example = solution(similar_matrix, prev_example, limit_num)
            i += 1
        str_current_example = ''.join(str(e) for e in current_example)

        for k in range(len(str_current_example)):
            a = str_current_example[k]
            b = example[k]
            if str_current_example[k] == example[k]:
                sum1 += 1

        # f = 0
        # while f < (len(str_current_example)):
        #     print(str_current_example[f:f + 10])
        #     f += 10

        percent = (sum1/len_of_example) * 100
        # print('%.3f' % percent + '%')
        if percent >= 98:
            counter += 1
    print('num of correct example: %d' % counter)


if __name__ == '__main__':
    main()

