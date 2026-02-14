// stencil.cu
#include <cuda_runtime.h>
#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output,
                               unsigned int n, unsigned int R) {
    extern __shared__ float s[];
    float* s_mask = s;                          // (2R+1)
    float* s_img  = s_mask + (2 * R + 1);       // (blockDim + 2R)
    float* s_out  = s_img  + (blockDim.x + 2 * R); // (blockDim)

    unsigned int tid = threadIdx.x;
    int base = (int)blockIdx.x * (int)blockDim.x;

    // load mask to shared
    if (tid < 2 * R + 1) s_mask[tid] = mask[tid];

    // load needed image (including halo) to shared
    int start = base - (int)R;
    for (unsigned int t = tid; t < blockDim.x + 2 * R; t += blockDim.x) {
        int g = start + (int)t;
        s_img[t] = (g < 0 || g >= (int)n) ? 1.0f : image[g];
    }
    __syncthreads();

    // compute one output element per thread
    int i = base + (int)tid;
    if (i < (int)n) {
        float sum = 0.0f;
        for (int j = -(int)R; j <= (int)R; j++)
            sum += s_img[tid + R + j] * s_mask[j + (int)R];
        s_out[tid] = sum;
    }
    __syncthreads();

    if (i < (int)n) output[i] = s_out[tid];
}

__host__ void stencil(const float* image, const float* mask, float* output,
                      unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t shmem = (size_t)( (2*R + 1) + (threads_per_block + 2*R) + threads_per_block ) * sizeof(float);
    stencil_kernel<<<blocks, threads_per_block, shmem>>>(image, mask, output, n, R);
}