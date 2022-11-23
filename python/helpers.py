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
    convert wacky number into up, right, down, left
    '''
    if n == 1:
        return 0
    if n == -1:
        return 2
    if n == 2:
        return 1
    if n == -2:
        return 3




def initialise_grid(width, height):
    # initialise grid with uniform probability
    grid = []
    for i in range(height):
        grid.append([])
        for j in range(width):
            grid[i].append(0)            
    return grid    


# def load_policy(file):
#     '''
#     Helper function to load the input.csv and set the frozen column
#     Takes relative filepath and returns grid, frozen_column
#     '''
#     with open(file, 'r', encoding='utf-8') as f:
#         grid = []
#         for row_number in range(0, 3):
#             line = f.readline().strip().split(',')
#             row = []
#             for column_number in range(0, 4):
#                 row.append(int_to_action(int(line[column_number])))
#             grid.append(row)
#         return grid

# Visualization


def print_grid(arr, policy=False):
    res = ""
    for row in range(3):
        res += "|"
        for column in range(4):
            if row == 1 and column == 3:
                val = "-1"
            elif row == 0 and column == 3:
                val = "+1"
            else:
                if policy:
                    val = ["Up", "Right", "Down", "Left"][arr[row][column]]
                else:
                    val = str(arr[row][column])
            if row == 1 and column == 1:
                val = "WALL"
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