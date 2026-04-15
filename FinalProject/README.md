# Parallel Jacobi Solver

This project implements a **serial and OpenMP parallel Jacobi solver** for solving linear systems:

Ax = b

The goal of the project is to evaluate the performance improvement of OpenMP parallelization compared to the serial implementation.

---

## Project Structure

```
src/        source code
scripts/    helper scripts for running experiments
results/    performance results and output data
```

---

## Compile

```bash
g++ -O2 src/jacobi_serial.cpp src/matrix_utils.cpp -o jacobi_serial
```

(OpenMP version)

```bash
g++ -O2 -fopenmp src/jacobi_omp.cpp src/matrix_utils.cpp -o jacobi_omp
```

---

## Run

Example with matrix size 512:

```bash
./jacobi_serial 512
```

or

```bash
./jacobi_omp 512
```

---

## Description

The Jacobi method is an iterative algorithm for solving systems of linear equations.  
In this project:

- A **random diagonally dominant matrix** is generated
- The system is solved using the **Jacobi iterative method**
- Performance is compared between **serial and OpenMP parallel implementations**

---

## Output

The program prints:

- matrix size
- number of iterations
- residual error
- runtime

Example:

```
Matrix size: 512
Iterations: 243
Residual: 1e-6
Time: 0.021 s
```
