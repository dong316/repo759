#!/bin/bash
#SBATCH --job-name=jacobi_serial
#SBATCH --output=results/%x_%j.out
#SBATCH --error=results/%x_%j.err
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

mkdir -p results

g++ -O2 src/jacobi_serial.cpp src/matrix_utils.cpp -o jacobi_serial

./jacobi_serial 256 10000 1e-6