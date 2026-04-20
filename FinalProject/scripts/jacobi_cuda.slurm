#!/bin/bash
#SBATCH --job-name=jacobi_cuda
#SBATCH --output=results/%x_%j.out
#SBATCH --error=results/%x_%j.err
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00

mkdir -p results

module load nvidia/cuda/13.0.0

nvcc -O2 -std=c++17 src/jacobi_cuda.cu src/matrix_utils.cpp -o jacobi_cuda

rm -f results/runtime_cuda.csv

for N in 256 512 1024 2048 4096
do
    for R in 1 2 3
    do
        ./jacobi_cuda $N 10000 1e-6 42
    done
done
