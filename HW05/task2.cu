// task2.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include "reduce.cuh"

int main(int argc, char **argv) {

    unsigned int N = std::stoi(argv[1]);
    unsigned int threads_per_block = std::stoi(argv[2]);

    // create host array with random values [-1,1]
    std::vector<float> h(N);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (unsigned int i = 0; i < N; i++)
        h[i] = dist(gen);

    // allocate device memory
    float *d_input;
    cudaMalloc(&d_input, N * sizeof(float));

    cudaMemcpy(d_input, h.data(),
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    // allocate output for first round
    unsigned int first_blocks =
        (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    float *d_output;
    cudaMalloc(&d_output,
               first_blocks * sizeof(float));

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce(&d_input, &d_output, N, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // copy result back
    float result;
    cudaMemcpy(&result, d_input,
               sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << result << "\n";
    std::cout << std::fixed
              << std::setprecision(3)
              << ms << "\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}