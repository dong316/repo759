#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include "matrix_utils.h"

using namespace std;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                 << " -> " << cudaGetErrorString(err) << endl;               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void jacobi_step_kernel(const double* A,
                                   const double* b,
                                   const double* x,
                                   double* x_new,
                                   double* err,
                                   int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; j++) {
            if (j != i) {
                sigma += A[i * N + j] * x[j];
            }
        }

        double new_val = (b[i] - sigma) / A[i * N + i];
        x_new[i] = new_val;

        atomicAdd(err, fabs(new_val - x[i]));
    }
}

__global__ void copy_kernel(double* x, const double* x_new, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] = x_new[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./jacobi_cuda N [max_iter] [tol] [seed]" << endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int max_iter = 10000;
    double tol = 1e-6;
    unsigned int seed = 42;

    if (argc >= 3) {
        max_iter = atoi(argv[2]);
    }
    if (argc >= 4) {
        tol = atof(argv[3]);
    }
    if (argc >= 5) {
        seed = (unsigned int) atoi(argv[4]);
    }

    // Host-side original system (reuse existing generator)
    vector<vector<double>> A_2d(N, vector<double>(N));
    vector<double> b(N);
    generate_diagonally_dominant_system(A_2d, b, N, seed);

    // Flatten A for GPU
    vector<double> A_flat(N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A_2d[i][j];
        }
    }

    vector<double> x_host(N, 0.0);
    vector<double> x_new_host(N, 0.0);

    // Device memory
    double* d_A = nullptr;
    double* d_b = nullptr;
    double* d_x = nullptr;
    double* d_x_new = nullptr;
    double* d_err = nullptr;

    size_t matrix_bytes = (size_t)N * (size_t)N * sizeof(double);
    size_t vector_bytes = (size_t)N * sizeof(double);

    CUDA_CHECK(cudaMalloc((void**)&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, vector_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_x, vector_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_x_new, vector_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_err, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_new, x_new_host.data(), vector_bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    int iter;
    double err = 0.0;

    auto start = chrono::high_resolution_clock::now();

    for (iter = 0; iter < max_iter; iter++) {
        err = 0.0;
        CUDA_CHECK(cudaMemcpy(d_err, &err, sizeof(double), cudaMemcpyHostToDevice));

        jacobi_step_kernel<<<blocks, threads_per_block>>>(d_A, d_b, d_x, d_x_new, d_err, N);
        CUDA_CHECK(cudaGetLastError());

        copy_kernel<<<blocks, threads_per_block>>>(d_x, d_x_new, N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(double), cudaMemcpyDeviceToHost));

        if (err < tol) {
            break;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end - start).count();

    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x, vector_bytes, cudaMemcpyDeviceToHost));

    // Residual L1 = sum_i |(Ax)_i - b_i|
    double residual = 0.0;
    for (int i = 0; i < N; i++) {
        double ax_i = 0.0;
        for (int j = 0; j < N; j++) {
            ax_i += A_flat[i * N + j] * x_host[j];
        }
        residual += fabs(ax_i - b[i]);
    }

    cout << fixed << setprecision(6);
    cout << "Jacobi CUDA solver finished" << endl;
    cout << "N = " << N << endl;
    cout << "Max iterations = " << max_iter << endl;
    cout << "Tolerance = " << tol << endl;
    cout << "Seed = " << seed << endl;
    cout << "Iterations used = " << iter + 1 << endl;
    cout << "Final error = " << err << endl;
    cout << "Runtime (s) = " << runtime << endl;
    cout << "Residual L1 = " << residual << endl;

    ofstream file("results/runtime_cuda.csv", ios::app);
    file << "cuda," << N << "," << iter + 1 << "," << runtime << endl;
    file.close();

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_new));
    CUDA_CHECK(cudaFree(d_err));

    return 0;
}