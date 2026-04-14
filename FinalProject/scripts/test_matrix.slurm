#!/bin/bash
#SBATCH --job-name=test_matrix
#SBATCH --output=results/test_matrix.out
#SBATCH --error=results/test_matrix.err
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

g++ -O2 src/test_matrix.cpp src/matrix_utils.cpp -o test_matrix

./test_matrix 8