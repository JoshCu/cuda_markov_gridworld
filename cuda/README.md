# Revisions  
The cuda files in this folder are different revisions with different optimisations.   

# Hardware  
CPU Intel i9-10850k base 3.6GHz, turbo 5.20 GHz, 10 cores 20 logical processors
GPU GeForce RTX 3080, Same speeds as reference GPU

## 0  
This contains the most basic implementation, takes the C serial version and just assigns one thread per grid location with no further optimisation.  
It's 10x slower than serial

## 1  
Added Tiling and some small logic optimisations, only uses global memory for operations, very expensive memory access