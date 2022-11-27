import random
import csv


def print_grid(arr, columns, rows, walls, wins, losses):
    res = ""
    for row in range(rows):
        res += "|"
        for column in range(columns):
            if (row, column) in walls:
                val = "WALL"
            elif (row, column) in wins:
                val = "WIN"
            elif (row, column) in losses:
                val = "LOSE"
            else:
                val = str(arr[row][column])

            res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)


def get_random_location(rows, columns, number, taken_locations):
    locations = []
    while len(locations) < number:
        new_location = (random.randint(0, rows - 1), random.randint(0, columns - 1))
        if new_location not in locations or new_location in taken_locations:
            locations.append(new_location)
    return locations


def create_grid(rows, columns, wins, losses, walls):
    grid = [[0 for i in range(columns)] for j in range(rows)]
    for wall in walls:
        grid[wall[0]][wall[1]] = "w"
    for loss in losses:
        grid[loss[0]][loss[1]] = -1
    for win in wins:
        grid[win[0]][win[1]] = 1
    return grid


def write_grid(value, file="case2.csv"):
    '''
    Helper function to save the output to csv
    '''
    wins = 0
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(value)


def make_grid(rows, columns, number_of_wins, number_of_losses, number_of_walls):
    taken_locations = []
    wins = get_random_location(rows, columns, number_of_wins, taken_locations)
    taken_locations += wins
    loses = get_random_location(rows, columns, number_of_losses, taken_locations)
    taken_locations += loses
    walls = get_random_location(rows, columns, number_of_walls, taken_locations)
    taken_locations += walls
    return create_grid(rows, columns, wins, loses, walls)


if __name__ == "__main__":

    for i in range(10, 101, 10):
        print(f"generating case {i}")
        for j in range(10):
            grid = make_grid(i, i, 1, 1, (i/2)**2)
            write_grid(grid, f"./small/case{i}_{j}.csv")

    for i in range(100, 501, 100):
        print(f"generating case {i}")
        for j in range(10):
            grid = make_grid(i, i, 1, 1, (i/2)**2)
            write_grid(grid, f"./large/case{i}_{j}.csv")
