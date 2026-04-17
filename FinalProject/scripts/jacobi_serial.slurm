#!/bin/bash
#SBATCH --job-name=jacobi_serial
#SBATCH --output=results/%x_%j.out
#SBATCH --error=results/%x_%j.err
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

mkdir -p results

g++ -O2 src/jacobi_serial.cpp src/matrix_utils.cpp -o jacobi_serial

for N in 256 512 1024 2048 4096 8192
do
    for i in 1 2 3
    do
        ./jacobi_serial $N
    done
done
