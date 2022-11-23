# Grid World in CUDA  
This repo contains the code that was written to evaluate a cuda implementation of the markov decision process.  
More specifically the use of value itteration to calculate the optimal policy for navigating a grid world.

## This repo is broken up into the following sections  

### Cases  
This is where all the different cases that were tested are stored.  
The files contain the utility for each square in a grid world.  
This folder also contains the script used to generate worlds of any dimension.  

### python  
This contains a python implementation of value itteration.
As it is python, it is quite slow. It's main purpose is to compare the results as it's a known good solution to the grid world problem.  

### c_serial  
This contains a serial implementation written in C for using in speedup comparisons.  

### cuda  
This contains the cuda implementation.