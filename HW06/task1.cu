#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mmul.h"

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n n_tests\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int n_tests = std::atoi(argv[2]);

    size_t size = n * n * sizeof(float);

    float *A, *B, *C;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // initialize in [-1,1]
    for (int i = 0; i < n*n; i++) {
        A[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        B[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        C[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    // ---- warm-up ----
    mmul(handle, A, B, C, n);
    cudaDeviceSynchronize();

    // ---- timing ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < n_tests; i++) {
        mmul(handle, A, B, C, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    float avg_ms = total_ms / n_tests;

    std::cout << avg_ms << std::endl;

    // cleanup
    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}