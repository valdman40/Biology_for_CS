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


def change_0_to_1(arr):
    indices_zero = arr == 0
    arr[indices_zero] = -1  # replacing 0s with -1
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


# change n percent of the array
def get_after_change_n_percent(arr, n=10):
    arr_copy = arr.copy()
    list_of_numbers = get_list_of_increasing_numbers(0, len(arr_copy))
    for _ in range(n):
        rand: int = random.choice(list_of_numbers)
        list_of_numbers.remove(rand)
        if arr_copy[rand] == 0:
            arr_copy[rand] = 1
        else:
            arr_copy[rand] = 0
    return arr_copy


def run_simulation(txt_filename, txt_prefect_digit, num_of_digits, number_of_runs, percent_to_pass_for_success, plot,
                   mix_up_percent):
    digits_line_by_line = get_digits_from_txt(txt_filename, num_of_digits)

    # digits_line_by_line = change_0_to_1(digits_line_by_line)

    perfect_digit = get_digits_from_txt(txt_prefect_digit, 1)[0]

    weight_matrix = get_weight_matrix(digits_line_by_line)

    digit_model = digits_line_by_line[0]
    digit_model = perfect_digit
    if plot:
        plt.imshow(digit_model.reshape(10, 10), aspect="auto")
        plt.title("digit model")
        plt.show()

    number_of_success = 0
    for _ in range(number_of_runs):
        digit_given = get_after_change_n_percent(digit_model, mix_up_percent)
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
    return success_rate * 100


def plot_3d_for_report(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z)
    ax.plot(x, y, z)
    plt.xlabel('number of letters', fontsize=10)
    plt.ylabel('mix upd percent', fontsize=10)
    ax.set_zlabel('success rate', fontsize=10)
    plt.show()


def get_txt_dig_name(method_index: int):
    return {
        0: f'zero_perfect.txt',
        1: f'one_perfect.txt',
        2: f'two_perfect.txt',
        3: f'three_perfect.txt',
        4: f'four_perfect.txt',
        5: f'five_perfect.txt',
        6: f'six_perfect.txt',
        7: f'seven_perfect.txt',
        8: f'eight_perfect.txt',
        9: f'nine_perfect.txt',
    }.get(method_index, 0)


def get_digit_to_find_from_user_input():
    while True:
        print('please choose digit you wish to play with')
        user_input = input()
        try:
            user_input = int(user_input)
            if user_input > 9 or user_input < 0:
                print('not valid digit')
            else:
                break
        except:
            print('please enter valid input 0-9')
    digit = get_txt_dig_name(user_input)
    return digit


def get_num_of_dig_from_user_input():
    while True:
        print('please how much different digits would you like to insert?')
        user_input = input()
        try:
            user_input = int(user_input)
            if user_input > 10 or user_input < 1:
                print('not valid number')
            else:
                break
        except:
            print('please enter valid input 1-10')
    amount = int(user_input)
    return amount


def get_plot_from_user_input():
    while True:
        print('would you like to view the digits output? 1-> yes, 0->no')
        user_input = input()
        try:
            user_input = int(user_input)
            if user_input > 1 or user_input < 0:
                print('not valid number')
            else:
                break
        except:
            print('please enter valid input 0/1')
    retval = False
    if user_input == 1:
        retval = True
    return retval


def main():
    txt_filename = '1_perfect_digit_of_each.txt'
    # txt_prefect_digit = 'zero_perfect.txt'
    txt_prefect_digit = get_digit_to_find_from_user_input()
    # hyper parameters
    num_of_different_digits = get_num_of_dig_from_user_input()
    num_of_digits = num_of_different_digits
    number_of_runs = 100
    percent_to_pass_for_success = 90
    mix_up_percent = 10
    plot_digits = get_plot_from_user_input()

    success_rate = run_simulation(txt_filename, txt_prefect_digit, num_of_digits, number_of_runs,
                                  percent_to_pass_for_success, plot_digits,
                                  mix_up_percent)
    print('for %d letters, success rate: %d%%' % (num_of_different_digits, success_rate))

    print('press somethong to exit')
    input()
    # x = []
    # y = []
    # z = []
    # ar = [5, 10, 20, 50]
    # for mix_up_percent in ar:
    #     print('--------mix up percent =  %d--------' % mix_up_percent)
    #     for num in range(1, num_of_different_digits + 1):
    #         success_rate = run_simulation(txt_filename, txt_prefect_digit, num * 10, number_of_runs,
    #                                       percent_to_pass_for_success, plot_digits,
    #                                       mix_up_percent)
    #         z.append(success_rate)
    #         x.append(num)
    #         y.append(mix_up_percent)
    #         print('for %d letters, success rate: %d%%' % (num, success_rate))
    # for num in range(1, num_of_different_digits):
    #     success_rate = run_simulation(txt_filename, txt_prefect_digit, num * 10, number_of_runs,
    #                                   percent_to_pass_for_success, plot_digits,
    #                                   mix_up_percent)
    #     x.append(success_rate)
    #     y.append(num)
    #     print('for %d letters, success rate: %d%%' % (num, success_rate))
    # #
    # # plot_3d_for_report(x, y, z)
    # fig = plt.figure()
    # plt.plot(y, x)
    # fig.suptitle('Check for learning 0', fontsize=16)
    # plt.xlabel('success rate', fontsize=10)
    # plt.ylabel('number of letters', fontsize=10)
    # plt.show()


if __name__ == '__main__':
    main()
