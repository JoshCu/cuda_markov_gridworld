"""
This file contains some helpers to load the policy and display the utility grid
   as well as dealing with the translation of notation for policy action direction
"""

__author__ = 'Josh Cunningham'
__copyright__ = 'Copyright 2022, MDP'
__email__ = 'Josh.Cu@gmail.com'

import csv


def int_to_action(n):
    '''
    convert number into up, right, down, left
    '''
    if n == 1:
        return 0
    if n == -1:
        return 2
    if n == 2:
        return 1
    if n == -2:
        return 3


def initialise_grid(width, height, value=0):
    # initialise grid with uniform probability
    grid = []
    for i in range(height):
        grid.append([])
        for j in range(width):
            grid[i].append(value)
    return grid


def load_utility(file):
    '''
    Helper function to load the input.csv and set the frozen column
    Takes relative filepath and returns grid, frozen_column
    '''
    # looping through the file twice is not ideal but it works
    # read first line in file as a string
    with open(file, "r") as f:
        first_line = f.readline()
        columns = len(first_line.split(","))
        rows = sum(1 for line in f) + 1
    walls = []
    endstates = []
    with open(file, 'r', encoding='utf-8') as f:
        grid = []
        for line_number in range(rows):
            line = f.readline().strip().split(',')
            row = []
            for column_number in range(columns):
                if line[column_number] == "w":
                    walls.append((line_number, column_number))
                    row.append(0)
                else:
                    row.append(int(line[column_number]))
                    if int(line[column_number]) != 0:
                        endstates.append((line_number, column_number))
            grid.append(row)
        return grid, columns, rows, walls, endstates

# Visualization


def print_grid(arr, model, policy=False):
    res = ""
    for row in range(model.num_rows):
        res += "|"
        for column in range(model.num_columns):
            if (row, column) in model.walls:
                val = "WALL"
            elif (row, column) in model.endstates:
                if model.current_utility[row][column] == 1:
                    val = "WIN"
                else:
                    val = "LOSE"
            else:
                if policy:
                    val = ["Up", "Right", "Down", "Left"][arr[row][column]]
                else:
                    val = str(arr[row][column])

            res += " " + val[:5].ljust(5) + " |"  # format
        res += "\n"
    print(res)


# def assignment_out(array):
#     output = []
#     for row in array:
#         r = []
#         for item in row:
#             r.append(action_to_int(item))
#         output.append(r)
#     print(output)
#     return output


def write_policy(value, file="expectimax.csv"):
    '''
    Helper function to save the output to csv
    '''
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(value)
