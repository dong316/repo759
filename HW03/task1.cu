#include <iostream>
#include <cuda_runtime.h>

// GPU kernel
__global__ void factorialKernel(int *dA)
{
    int a = threadIdx.x + 1;   // threadIdx.x = 0..7 → 1..8

    int fact = 1;
    for (int i = 1; i <= a; ++i)
    {
        fact *= i;
    }

    dA[a - 1] = fact;  // store in a-th entry (index a-1)
}

int main()
{
    const int N = 8;

    int hA[N];        // host array
    int *dA;          // device pointer

    // allocate device memory
    cudaMalloc((void**)&dA, N * sizeof(int));

    // launch kernel: 1 block, 8 threads
    factorialKernel<<<1, 8>>>(dA);

    // wait for GPU to finish
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    // print results (one per line)
    for (int i = 0; i < N; ++i)
    {
        std::cout << hA[i] << std::endl;
    }

    // free device memory
    cudaFree(dA);

    return 0;
}