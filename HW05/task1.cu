#include "matmul.cuh"
#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

static inline void cudaCheck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(e)
              << "\n";
    throw std::runtime_error("CUDA failure");
  }
}

template <typename T>
static void fill_matrices(std::vector<T> &A, std::vector<T> &B, unsigned int n) {
  // "Fill however you like" — keep it deterministic.
  // A: A[i,j] = (i+j) % 7
  // B: B[i,j] = (i-j) % 5
  for (unsigned int i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < n; ++j) {
      A[i * n + j] = (T)((i + j) % 7);
      B[i * n + j] = (T)(((int)i - (int)j) % 5);
    }
  }
}

template <typename T>
static void run_one(void (*matmul_fn)(const T *, const T *, T *, unsigned int,
                                      unsigned int),
                    unsigned int n, unsigned int block_dim) {
  const size_t bytes = (size_t)n * (size_t)n * sizeof(T);

  std::vector<T> hA(n * n), hB(n * n), hC(n * n);
  fill_matrices(hA, hB, n);

  T *dA = nullptr;
  T *dB = nullptr;
  T *dC = nullptr;

  cudaCheck(cudaMalloc((void **)&dA, bytes), "cudaMalloc A");
  cudaCheck(cudaMalloc((void **)&dB, bytes), "cudaMalloc B");
  cudaCheck(cudaMalloc((void **)&dC, bytes), "cudaMalloc C");

  cudaCheck(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice),
            "H2D A");
  cudaCheck(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice),
            "H2D B");

  // time with CUDA events (ms)
  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start), "event create start");
  cudaCheck(cudaEventCreate(&stop), "event create stop");

  cudaCheck(cudaEventRecord(start), "event record start");
  matmul_fn(dA, dB, dC, n, block_dim);
  cudaCheck(cudaEventRecord(stop), "event record stop");
  cudaCheck(cudaEventSynchronize(stop), "event sync stop");

  float ms = 0.0f;
  cudaCheck(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

  cudaCheck(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost),
            "D2H C");

  // Print: first element, last element, time (ms)
  if constexpr (std::is_same_v<T, int>) {
    std::cout << hC[0] << "\n";
    std::cout << hC[n * n - 1] << "\n";
    std::cout << std::fixed << std::setprecision(1) << ms << "\n";
  } else {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << (double)hC[0] << "\n";
    std::cout << (double)hC[n * n - 1] << "\n";
    std::cout << ms << "\n";
  }

  cudaCheck(cudaEventDestroy(start), "event destroy start");
  cudaCheck(cudaEventDestroy(stop), "event destroy stop");

  cudaCheck(cudaFree(dA), "cudaFree A");
  cudaCheck(cudaFree(dB), "cudaFree B");
  cudaCheck(cudaFree(dC), "cudaFree C");
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: ./task1 n block_dim\n";
    return 1;
  }

  unsigned int n = (unsigned int)std::strtoul(argv[1], nullptr, 10);
  unsigned int block_dim = (unsigned int)std::strtoul(argv[2], nullptr, 10);

  if (n == 0 || block_dim == 0) {
    std::cerr << "Error: n and block_dim must be positive.\n";
    return 1;
  }

  // The kernel assumes square blocks block_dim x block_dim
  // Typical choices: 8/16/32
  if (block_dim > 32) {
    std::cerr << "Warning: block_dim > 32 may be slow or exceed resources.\n";
  }

  // int
  run_one<int>(matmul_1, n, block_dim);
  // float
  run_one<float>(matmul_2, n, block_dim);
  // double
  run_one<double>(matmul_3, n, block_dim);

  return 0;
}
