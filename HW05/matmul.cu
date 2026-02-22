#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

static inline void cudaCheck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    throw std::runtime_error("CUDA failure");
  }
}

template <typename T>
__global__ void matmul_kernel_tiled(const T *A, const T *B, T *C,
                                   unsigned int n) {
  // 2D thread indices
  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned int TILE = blockDim.x; // assume square block_dim x block_dim

  // Dynamic shared memory layout: [As | Bs]
  extern __shared__ unsigned char smem[];
  T *As = reinterpret_cast<T *>(smem);
  T *Bs = reinterpret_cast<T *>(smem + TILE * TILE * sizeof(T));

  const unsigned int ty = threadIdx.y;
  const unsigned int tx = threadIdx.x;

  T acc = (T)0;

  // Number of tiles along K dimension
  const unsigned int numTiles = (n + TILE - 1) / TILE;

  for (unsigned int t = 0; t < numTiles; ++t) {
    // Global indices to load
    const unsigned int aCol = t * TILE + tx; // k
    const unsigned int bRow = t * TILE + ty; // k

    // Load A tile element (row, aCol)
    if (row < n && aCol < n) {
      As[ty * TILE + tx] = A[row * n + aCol];
    } else {
      As[ty * TILE + tx] = (T)0;
    }

    // Load B tile element (bRow, col)
    if (bRow < n && col < n) {
      Bs[ty * TILE + tx] = B[bRow * n + col];
    } else {
      Bs[ty * TILE + tx] = (T)0;
    }

    __syncthreads();

    // Multiply-accumulate this tile
    #pragma unroll
    for (unsigned int k = 0; k < TILE; ++k) {
   	 acc += As[ty * TILE + k] * Bs[k * TILE + tx];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = acc;
  }
}

template <typename T>
static void matmul_dispatch(const T *A, const T *B, T *C, unsigned int n,
                            unsigned int block_dim) {
  dim3 block(block_dim, block_dim);
  dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

  size_t shmem = 2ULL * block_dim * block_dim * sizeof(T);

  matmul_kernel_tiled<T><<<grid, block, shmem>>>(A, B, C, n);
  cudaCheck(cudaGetLastError(), "kernel launch");
  cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim) {
  matmul_dispatch<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
  matmul_dispatch<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
  matmul_dispatch<double>(A, B, C, n, block_dim);
}
