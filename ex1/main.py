from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# setting up the values for the grid
ON = 255
OFF = 0
vals = [ON, OFF]

EMPTY = 0
HEALTHY = 1
SICK = 2
HEALTHY_IMMUNE = 3

P_HEALTHY_INFECTED_BY_SICK = 0.6
P_HEALTHY_IMUNNED_INFECTED_BY_SICK = 0.2
T = 10  # generation limit

# classifications_colors = ['white', 'green', 'red', 'blue']
classifications_colors = ['w', 'g', 'r', 'b']
classifications_values = [EMPTY, HEALTHY, SICK, HEALTHY_IMMUNE]


def pick_2_random_numbers_in_range(N):
    # Request two random integers between 0 and 3 (exclusive)
    indices = np.random.randint(0, high=N, size=2)

    # Extract the row and column indices
    i = indices[0]
    j = indices[1]
    return i, j


def put_value_in_empty_grid_cell(grid, value):
    i, j = pick_2_random_numbers_in_range(len(grid))
    # keeps trying to get empty cell to fill
    while grid[i, j] != 0:
        i, j = pick_2_random_numbers_in_range(len(grid))
    grid[i, j] = value


def fill_value_in_grid(grid, count, value):
    for i in range(count):
        put_value_in_empty_grid_cell(grid, value)


def generate_random_population_map(world_size, healthy_count, sick_count, healthy_immune_count):
    empty_count = (world_size * world_size) - (healthy_count + sick_count + healthy_immune_count)
    if empty_count < 0:
        raise Exception("Too much population for the size of map you have chosen")

    population_map = np.zeros(shape=(world_size, world_size))  # init map
    # fill grid with population
    fill_value_in_grid(population_map, healthy_count, HEALTHY)
    fill_value_in_grid(population_map, sick_count, SICK)
    fill_value_in_grid(population_map, healthy_immune_count, HEALTHY_IMMUNE)
    return population_map


def get_next_possible_moves(i, j, N):
    right = (i + 1) % N
    left = i - 1
    if left < 0:
        left = N - 1
    down = (j + 1) % N
    up = j - 1
    if up < 0:
        up = N - 1
    neighbors_position = [
        [left, up],
        [i, up],
        [right, up],
        [right, j],
        [right, down],
        [i, down],
        [left, down],
        [left, j],
        [i, j],
    ]
    return neighbors_position


LEFT_UP = 0
UP = 1
RIGHT_UP = 2
RIGHT = 3
RIGHT_DOWN = 4
DOWN = 5
LEFT_DOWN = 6
LEFT = 7
CURRENT = 8


def get_person_health(world_map, i, j, neighbors_position):
    current_health = world_map[i][j]
    # check if have sick person around me
    sick_near_me = False
    for neighbor_pos in neighbors_position:
        n_i, n_j = neighbor_pos[0], neighbor_pos[1]
        neighbor_health = world_map[n_i][n_j]
        if neighbor_health == SICK:
            sick_near_me = True
    if sick_near_me:
        if current_health == HEALTHY:
            current_health = np.random.choice([HEALTHY, SICK],
                                              p=[1 - P_HEALTHY_INFECTED_BY_SICK, P_HEALTHY_INFECTED_BY_SICK])
        elif current_health == HEALTHY_IMMUNE:
            current_health = np.random.choice([HEALTHY_IMMUNE, SICK],
                                              p=[1 - P_HEALTHY_IMUNNED_INFECTED_BY_SICK,
                                                 P_HEALTHY_IMUNNED_INFECTED_BY_SICK])
    return current_health


def update(frameNum, img, world_map, N):
    new_world_map = world_map.copy()
    for i in range(N):
        for j in range(N):
            current_value = world_map[i][j]
            if current_value != EMPTY:
                next_possible_move = get_next_possible_moves(i, j, N)
                # decide new value for current cell
                current_value = get_person_health(world_map, i, j, next_possible_move)
                # moving to new position
                choosing_array = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                for k in range(8):
                    choice = np.random.choice(choosing_array)
                    next_pos = next_possible_move[choice]
                    choosing_array.remove(choice)
                    next_i, next_j = next_pos[0], next_pos[1]
                    if new_world_map[next_i][next_j] == EMPTY:
                        new_world_map[next_i][next_j] = current_value
                        new_world_map[i][j] = EMPTY
                        break

    # update data
    img.set_data(new_world_map)
    world_map[:] = new_world_map[:]
    return img,

def main():
    # set grid size
    N = 50
    Nh = 200  # healthy
    Ns = 1  # sick
    Nv = 100  # immune

    # set animation update interval
    updateInterval = 10

    # declare grid
    population_map = generate_random_population_map(world_size=N,
                                                    healthy_count=Nh,
                                                    sick_count=Ns,
                                                    healthy_immune_count=Nv)

    # set up animation
    figure, axes = plt.subplots()
    cmap = ListedColormap(classifications_colors)
    img = axes.imshow(population_map, interpolation='nearest', cmap=cmap)
    ani = animation.FuncAnimation(figure, update, fargs=(img, population_map, N,),
                                  frames=T,
                                  interval=updateInterval,
                                  save_count=50,
                                  repeat=False)
    plt.show()
    input()


if __name__ == '__main__':
    main()
