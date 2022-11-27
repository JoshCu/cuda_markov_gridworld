// hello world in c
#include <stdio.h>
#include <math.h>
#include <time.h>

#define PENALTY -0.04
#define DISCOUNT 0.95
#define MAX_ITER 1000
#define NUM_ACTIONS 4
#define SUCCESS_CHANCE 0.8

void loadGrid(FILE *fp, float *grid, int rows, int cols);
void printGrid(float *grid, int rows, int cols);
float getBestValue(float *grid, int rows, int cols, int x, int y);

struct State
{
    int x;
    int y;
};

State actions[NUM_ACTIONS] = {
    {-1, 0},  // up
    {0, 1},   // right
    {1, 0},   // down
    {0, -1}}; // left

int main(int argc, char *argv[])
{
    struct timespec startOverhead, startComputation, endOverhead, endComputation;
    clock_gettime(CLOCK_MONOTONIC, &startOverhead);
    char c; // char to hold the input

    // get file input from args
    if (argc != 2)
    {
        printf("filename argument expected.\n");
        exit(1);
    }

    // Load in file
    FILE *fp;
    fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("Error opening file");
        return 1;
    }
    // get grid dimensions from file
    int rows, cols, walls, endstates;
    rows = cols = walls = endstates = 0;
    for (c = getc(fp); c != EOF; c = getc(fp))
    {
        if (c == '\n') // Increment count if this character is newline
            rows += 1;
        if (c == ',')
            cols += 1;
        if (c == 'w')
            walls += 1;
        if (c == '1')
            endstates += 1;
    }
    cols = cols / rows;
    cols += 1; // add one for the last column
    printf("rows: %d, cols: %d\n", rows, cols);
    printf("walls: %d\n", walls);

    // reset file pointer
    rewind(fp);

    // create matrix of grid dimensions
    float grid[rows][cols];

    // load grid from csv into matrix
    loadGrid(fp, grid[0], rows, cols);

    // close file
    fclose(fp);

    // pass in grid to print function as a pointer
    // printGrid(grid[0], rows, cols);

    clock_gettime(CLOCK_MONOTONIC, &endOverhead);
    clock_gettime(CLOCK_MONOTONIC, &startComputation);

    // bellman equation
    float newGrid[rows][cols];
    float maxChange = 0;
    float change = 0;
    int iter = 0;
    while (iter < MAX_ITER)
    {
        maxChange = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (grid[i][j] == 1)
                {
                    newGrid[i][j] = 1;
                    continue;
                }
                if (grid[i][j] == -1)
                {
                    newGrid[i][j] = -1;
                    continue;
                }
                if (grid[i][j] == 2)
                {
                    newGrid[i][j] = 2;
                    continue;
                }
                float bestValue = getBestValue(grid[0], rows, cols, i, j);
                newGrid[i][j] = bestValue;
                change = fabs(bestValue - grid[i][j]);
                if (change > maxChange)
                {
                    maxChange = change;
                }
            }
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                grid[i][j] = newGrid[i][j];
            }
        }
        iter++;
        if (maxChange < 0.0001)
        {
            break;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &endComputation);
    double time_taken;
    time_taken = (endOverhead.tv_sec - startOverhead.tv_sec) * 1e9;
    time_taken = (time_taken + (endOverhead.tv_nsec - startOverhead.tv_nsec)) * 1e-9;
    printf("Overhead Time: %f\n", time_taken);
    time_taken = (endComputation.tv_sec - startComputation.tv_sec) * 1e9;
    time_taken = (time_taken + (endComputation.tv_nsec - startComputation.tv_nsec)) * 1e-9;
    printf("Computation Time: %f\n", time_taken);
    time_taken = (endComputation.tv_sec - startOverhead.tv_sec) * 1e9;
    time_taken = (time_taken + (endComputation.tv_nsec - startOverhead.tv_nsec)) * 1e-9;
    printf("Total Time: %f\n", time_taken);
    // printGrid(grid[0], rows, cols);
    return 0;
}

// check values of surrounding cells and return the best value
float getBestValue(float *grid, int rows, int cols, int i, int j)
{
    float bestValue = -100;
    for (int a = 0; a < NUM_ACTIONS; a++)
    {
        State action = actions[a];
        State s = {i, j};
        State s2 = {i + action.x, j + action.y};
        if (s2.x < 0 || s2.x >= rows || s2.y < 0 || s2.y >= cols || *(grid + s2.x * cols + s2.y) == 2) // grid[s2.x][s2.y] == 2)
        {
            s2 = s;
        }
        State actionLeft = actions[(a - 1) % NUM_ACTIONS];
        State s3 = {i + actionLeft.x, j + actionLeft.y};
        if (s3.x < 0 || s3.x >= rows || s3.y < 0 || s3.y >= cols || *(grid + s3.x * cols + s3.y) == 2)
        {
            s3 = s;
        }
        State actionRight = actions[(a + 1) % NUM_ACTIONS];
        State s4 = {i + actionRight.x, j + actionRight.y};
        if (s4.x < 0 || s4.x >= rows || s4.y < 0 || s4.y >= cols || *(grid + s4.x * cols + s4.y) == 2)
        {
            s4 = s;
        }
        float slip_chance = (1 - SUCCESS_CHANCE) / 2;
        float v = SUCCESS_CHANCE * (PENALTY + DISCOUNT * *(grid + s2.x * cols + s2.y));
        v += slip_chance * (PENALTY + DISCOUNT * *(grid + s3.x * cols + s3.y));
        v += slip_chance * (PENALTY + DISCOUNT * *(grid + s4.x * cols + s4.y));
        if (v > bestValue)
        {
            bestValue = v;
        }
    }
    return bestValue;
}

// load grid function
void loadGrid(FILE *fp, float *grid, int rows, int cols)
{
    char c;
    int i = 0, j = 0;
    int skip = 0;
    for (c = getc(fp); c != EOF; c = getc(fp))
    {
        if (c == '\n')
        {
            i++;
            j = 0;
        }
        else if (c == ',')
        {
            j++;
        }
        else if (c == 'w')
        {
            // if wall, set to 2 as utility can never be higher than 1
            *(grid + i * cols + j) = 2;
        }
        else if (c == '-')
        {
            *(grid + i * cols + j) = -1;
            skip = 1;
        }
        else
        {
            if (!skip)
                *(grid + i * cols + j) = c - '0';
            skip = 0;
        }
    }
}

// print grid using pointer
void printGrid(float *grid, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", *(grid + i * cols + j));
        }
        printf("\n");
    }
    printf("\n");
}
