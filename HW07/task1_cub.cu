#define CUB_STDERR
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"

using namespace cub;

CachingDeviceAllocator g_allocator(true);

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./task1_cub n\n");
        return 1;
    }

    int n = atoi(argv[1]);

    // host array
    float* h_in = (float*)malloc(sizeof(float) * n);

    for (int i = 0; i < n; i++) {
        h_in[i] = -1.0f + 2.0f * rand() / RAND_MAX;
    }

    // device input
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * n));

    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    // device output
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_sum, sizeof(float)));

    // temp storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // actual reduction
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", gpu_sum);
    printf("%f\n", ms);

    // cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    free(h_in);

    return 0;
}