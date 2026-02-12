// task2.cu
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <random>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                    \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__global__ void fill(int* dA, int a) {
  int x = threadIdx.x;                 // threadIdx
  int y = blockIdx.x;                  // blockIdx
  int idx = y * blockDim.x + x;        // distinct entry in dA for each thread

  dA[idx] = a * x + y;                 // compute a*x + y
}

int main() {
  constexpr int N = 16;

  // 1) allocate device array dA
  int* dA = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, N * sizeof(int)));

  // 2) generate a using "The C++ Way": fixed seed + mt19937
  int some_seed = 759;                 // fixed seed (reproducible)
  std::mt19937 generator(some_seed);   // Mersenne Twister engine
  std::uniform_int_distribution<int> dist(1, 20);
  int a = dist(generator);

  // 3) launch kernel: 2 blocks, 8 threads each
  fill<<<2, 8>>>(dA, a);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4) copy back to host array hA
  int hA[N];
  CUDA_CHECK(cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost));

  // 5) print 16 values separated by single spaces
  for (int i = 0; i < N; ++i) {
    std::cout << hA[i] << (i == N - 1 ? '\n' : ' ');
  }

  CUDA_CHECK(cudaFree(dA));
  return 0;
}