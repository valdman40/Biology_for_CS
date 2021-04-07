import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    matrix_size = 500
    classifications = ['k', 'w', 'r'] # different classes
    # k - 0 - black
    # w - 1 - white
    # r - 2 - red
    matrix = np.random.randint(len(classifications), size=(matrix_size, matrix_size))
    print(matrix)
    cmap = ListedColormap(['k', 'w', 'r'])
    cax = plt.matshow(matrix, cmap=cmap)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
