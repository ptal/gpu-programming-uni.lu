#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --reservation=GPU_Programming_10d2
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=slurm.out

# Load the CUDA compiler.
module load system/CUDA

# Compile the code.
nvcc hello.cu -arch=sm_70 -std=c++17 -O3 -o hello

# Execute the code. Anything output will be stored in slurm.out (see above).
./hello
