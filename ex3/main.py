import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


def get_digits_from_txt(filename, num_of_letters):
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


def get_weight_matrix(digits_line_by_line):
    col_size = digits_line_by_line.shape[1]
    weight_matrix = np.zeros(shape=(col_size, col_size))
    for i in range(col_size):
        weight_matrix[i][i] = col_size + 1
        col = digits_line_by_line[:, i]
        for j in range(col_size):
            # means it's same cell ([i][i]), so we already did it above
            if i == j:
                continue
            second_col = digits_line_by_line[:, j]
            sum_of_similar = 0
            for k in range(len(col)):
                if col[k] == second_col[k]:
                    sum_of_similar += 1
            sum_of_different = len(col) - sum_of_similar
            weight = sum_of_similar - sum_of_different
            weight_matrix[i][j] = weight
    return weight_matrix


# returns list of numbers in increasing order
def get_list_of_increasing_numbers(start: int, end: int):
    list_of_numbers: List[int] = []
    for i in range(start, end):
        list_of_numbers.append(i)
    return list_of_numbers


def update(weight_matrix, example):
    sum1 = 0
    list_of_numbers = get_list_of_increasing_numbers(0, len(example))
    copy_example = example.copy()
    while len(list_of_numbers) > 0:
        place: int = random.choice(list_of_numbers)
        list_of_numbers.remove(place)
        weight_column = weight_matrix[:, place]
        for i in range(len(weight_column)):
            if i != place:
                sum1 += weight_column[i] * copy_example[i]
        if sum1 >= 0:
            grade = 1
        else:
            grade = 0
        copy_example[place] = grade
        sum1 = 0

    return copy_example


# change 10 percent of the array
def get_after_change_10_percent(arr):
    arr_copy = arr.copy()
    list_of_numbers = get_list_of_increasing_numbers(0, len(arr_copy))
    for _ in range(int(len(arr_copy) / 10)):
        rand: int = random.choice(list_of_numbers)
        list_of_numbers.remove(rand)
        if arr_copy[rand] == 0:
            arr_copy[rand] = 1
        else:
            arr_copy[rand] = 0
    return arr_copy


def run_simulation(txt_filename, num_of_digits, number_of_runs, percent_to_pass_for_success, plot):
    digits_line_by_line = get_digits_from_txt(txt_filename, num_of_digits)

    weight_matrix = get_weight_matrix(digits_line_by_line)

    digit_model = digits_line_by_line[0]
    if plot:
        plt.imshow(digit_model.reshape(10, 10), aspect="auto")
        plt.title("digit model")
        plt.show()

    number_of_success = 0
    for _ in range(number_of_runs):
        digit_given = get_after_change_10_percent(digit_model)
        if plot:
            plt.imshow(digit_given.reshape(10, 10), aspect="auto")
            plt.title("digit after mix up")
            plt.show()
        digit_updated = update(weight_matrix, digit_given)
        # while not convergence
        while not (digit_updated == digit_given).all():
            digit_given = digit_updated
            digit_updated = update(weight_matrix, digit_given)
        if plot:
            plt.imshow(digit_updated.reshape(10, 10), aspect="auto")
            plt.title("digit after update")
            plt.show()
        total_matches = np.sum(digit_updated == digit_model)
        percent = (total_matches / len(digit_model)) * 100
        # print("round num %d, percent of success = %d%%" % (_ + 1, percent))
        if percent >= percent_to_pass_for_success:
            number_of_success += 1

    success_rate = number_of_success / number_of_runs
    print('success rate: %d%%' % (success_rate * 100))
    return 0, 0, 0


def main():
    txt_filename = 'Digits.txt'

    # hyper parameters
    num_of_digits = 2
    number_of_runs = 100
    percent_to_pass_for_success = 98
    plot_digits = False

    x_axes = []  # num of digits
    y_axes = []  # rate of change
    z_axes = []  # success rate
    x, y, z = run_simulation(txt_filename, num_of_digits, number_of_runs, percent_to_pass_for_success, plot_digits)
    x_axes.append(x)
    y_axes.append(y)
    z_axes.append(z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_axes, y_axes, z_axes)
    # plt.show()


if __name__ == '__main__':
    main()
