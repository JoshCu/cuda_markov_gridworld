// hello world in c
#include <stdio.h>
#include <math.h>
#include <time.h>

#define PENALTY -0.04
#define DISCOUNT 0.95
#define MAX_ITER 1000
#define NUM_ACTIONS 4
#define SUCCESS_CHANCE 0.8
#define TILE_WIDTH 32
#define CHANGE_THRESHOLD 0.0001

void loadGrid(FILE *fp, float *grid, int rows, int cols);
void printGrid(float *grid, int rows, int cols);

__device__ float getBestValue(float *grid, int rows, int cols, int x, int y);
__global__ void bellman(float *grid, float *newGrid, int rows, int cols, int *changeFlag);
__device__ void dprintGrid(float *grid, int rows, int cols);

struct State
{
    int x;
    int y;
};

int main(int argc, char *argv[])
{
    clock_t startOverhead;
    startOverhead = clock();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
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
    // allocate the array
    float *grid = new float[rows * cols];

    // load grid from csv into matrix
    loadGrid(fp, grid, rows, cols);

    // close file
    fclose(fp);

    // pass in grid to print function as a pointer
    // printGrid(grid, rows, cols);
    printf("Overhead Time: %f\n", ((double)(clock() - startOverhead)) / CLOCKS_PER_SEC);

    // bellman equation
    float *newGrid = new float[rows * cols];

    // CUDA
    float *d_grid;
    float *d_newGrid;
    int size = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_grid, size);
    cudaMalloc((void **)&d_newGrid, size);
    cudaMemcpy(d_grid, grid, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_newGrid, newGrid, size, cudaMemcpyHostToDevice);

    // Call kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(ceil((float)rows / (float)TILE_WIDTH), ceil((float)cols / TILE_WIDTH));

    // print grid dimensions
    printf("grid rows: %f, grid cols: %f\n", ceil((float)rows / (float)TILE_WIDTH), ceil((float)cols / TILE_WIDTH));

    cudaEventRecord(start);
    // create a change flag
    int changeFlag = 1;
    int *d_changeFlag;
    cudaMalloc((void **)&d_changeFlag, sizeof(int));
    int iter = 0;
    while (changeFlag != 0 && iter < MAX_ITER)
    {
        changeFlag = 0;
        cudaMemset(d_changeFlag, 0, sizeof(int));
        bellman<<<dimGrid, dimBlock>>>(d_grid, d_newGrid, rows, cols, d_changeFlag);
        cudaDeviceSynchronize();
        cudaMemcpy(&changeFlag, d_changeFlag, sizeof(int), cudaMemcpyDeviceToHost);
        iter++;
    }

    cudaEventRecord(stop);

    // Copy back to host
    cudaMemcpy(grid, d_grid, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Computation Time: %f\n", milliseconds / 1000);
    printf("Total Time: %f\n", ((double)(clock() - startOverhead)) / CLOCKS_PER_SEC);

    // if (rows < 20)
    //     printGrid(grid, rows, cols);

    // print out the number of iterations
    printf("Iterations: %d\n", iter);

    // deallocate the array
    delete[] grid;
    delete[] newGrid;

    // free the device memory
    cudaFree(d_grid);
    cudaFree(d_changeFlag);

    return 0;
}

__global__ void bellman(float *grid, float *newGrid, int rows, int cols, int *changeFlag)
{

    float change = -1;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i * cols + j;
    __syncthreads();
    if (i < rows && j < cols)
    {
        if (grid[index] == 2 || grid[index] == 1 || grid[index] == -1)
        {
            newGrid[index] = grid[index];
            return;
        }

        // load edge values around tile into shared memory
        change = 0;
        __syncthreads();

        float bestValue = getBestValue(grid, rows, cols, i, j);
        newGrid[index] = bestValue;
        __syncthreads();
        // calculate change
        change = abs(abs(grid[index]) - abs(bestValue));
        // printf("index = %d  change: %f\n", index, change);
        __syncthreads();
        // if change is greater than max change, set max change to change
        if (change > CHANGE_THRESHOLD)
        {
            *changeFlag = 1;
        }
        __syncthreads();
        grid[index] = newGrid[index];
        __syncthreads();
    }
    // if (index == 0)
    // {
    //     dprintGrid(newGrid, rows, cols);
    //     printf("changeflag: %d\n", *changeFlag);
    // }
}

// check values of surrounding cells and return the best value
__device__ float getBestValue(float *grid, int rows, int cols, int i, int j)
{
    State actions[NUM_ACTIONS] = {
        {-1, 0},  // up
        {0, 1},   // right
        {1, 0},   // down
        {0, -1}}; // left
    float bestValue = -100;
    for (int a = 0; a < NUM_ACTIONS; a++)
    {
        State action = actions[a];
        State s = {i, j};
        State s2 = {i + action.x, j + action.y};
        if (s2.x < 0 || s2.x >= rows || s2.y < 0 || s2.y >= cols || grid[s2.x * cols + s2.y] == 2) // if out of bounds or wall
        {
            s2 = s;
        }
        State actionLeft = actions[(a - 1) % NUM_ACTIONS];
        State s3 = {i + actionLeft.x, j + actionLeft.y};
        if (s3.x < 0 || s3.x >= rows || s3.y < 0 || s3.y >= cols || grid[s3.x * cols + s3.y] == 2)
        {
            s3 = s;
        }
        State actionRight = actions[(a + 1) % NUM_ACTIONS];
        State s4 = {i + actionRight.x, j + actionRight.y};
        if (s4.x < 0 || s4.x >= rows || s4.y < 0 || s4.y >= cols || grid[s4.x * cols + s4.y] == 2)
        {
            s4 = s;
        }
        float slip_chance = (1 - SUCCESS_CHANCE) / 2;
        float v = SUCCESS_CHANCE * (PENALTY + DISCOUNT * grid[s2.x * cols + s2.y]);
        v += slip_chance * (PENALTY + DISCOUNT * grid[s3.x * cols + s3.y]);
        v += slip_chance * (PENALTY + DISCOUNT * grid[s4.x * cols + s4.y]);
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
            grid[i * cols + j] = 2;
        }
        else if (c == '-')
        {
            grid[i * cols + j] = -1;
            skip = 1;
        }
        else
        {
            if (!skip)
                grid[i * cols + j] = c - '0';
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
            printf("%f ", grid[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// print grid using pointer
__device__ void dprintGrid(float *grid, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", grid[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}