#!/usr/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=%x_%j.out
#SBATCH --export=ALL

# Load the CUDA compiler.
module load compiler/NVHPC/23.7-CUDA-12.1.1

# Manage the build environment.
mkdir --parent build

# Compile the code.
nvcc ${SLURM_JOB_NAME}.cu -arch=sm_70 -std=c++17 -O3 -o build/${SLURM_JOB_NAME}

# Execute the code. Anything output will be stored in %x_%j.out (`man sbatch`).
./build/${SLURM_JOB_NAME}

# Using a slurm variable to provide the name for the executable and source file
# is a hack and a fragile design. We can avoid this issue by using a build
# script such as Makefiles.
