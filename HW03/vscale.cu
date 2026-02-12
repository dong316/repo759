#include <cuda_runtime.h>
#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n) {

    // global thread index
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread performs at most one multiplication
    if (i < n) {
        b[i] = a[i] * b[i];
    }
}