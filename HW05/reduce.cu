// reduce.cu

#include <cuda_runtime.h>
#include "reduce.cuh"

// Kernel 4: First Add During Load
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {

    // dynamic shared memory
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    // each block handles 2 * blockDim elements
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    float sum = 0.0f;

    // first load
    if (i < n)
        sum = g_idata[i];

    // second load (first add during load)
    if (i + blockDim.x < n)
        sum += g_idata[i + blockDim.x];

    // store into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // reduce inside shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // write block result
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


// host function: repeatedly call kernel
__host__ void reduce(float **input, float **output,
                     unsigned int N,
                     unsigned int threads_per_block) {

    float *d_in  = *input;
    float *d_out = *output;

    unsigned int n = N;

    while (n > 1) {

        // number of blocks needed
        unsigned int blocks =
            (n + threads_per_block * 2 - 1) / (threads_per_block * 2);

        reduce_kernel<<<blocks,
                        threads_per_block,
                        threads_per_block * sizeof(float)>>>
                        (d_in, d_out, n);

        n = blocks;

        // swap input/output for next round
        float *temp = d_in;
        d_in  = d_out;
        d_out = temp;
    }

    // make sure final result is in original input
    if (d_in != *input) {
        cudaMemcpy(*input, d_in, sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();  // required for timing
}