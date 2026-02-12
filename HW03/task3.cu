// task3.cu
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include "vscale.cuh"

int main(int argc, char** argv) {
  unsigned int n = (argc > 1) ? (unsigned int)std::stoul(argv[1]) : 0;
  if (n == 0) return 1;

  std::vector<float> a(n), b(n);

  std::mt19937 gen(759);
  std::uniform_real_distribution<float> da(-10.0f, 10.0f), db(0.0f, 1.0f);
  for (unsigned int i = 0; i < n; ++i) { a[i] = da(gen); b[i] = db(gen); }

  float *dA, *dB;
  cudaMalloc(&dA, n * sizeof(float));
  cudaMalloc(&dB, n * sizeof(float));
  cudaMemcpy(dA, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const unsigned int TPB = 512;
  unsigned int blocks = (n + TPB - 1) / TPB;

  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  vscale<<<blocks, TPB>>>(dA, dB, n);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);

  cudaMemcpy(b.data(), dB, n * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << ms << "\n" << b[0] << "\n" << b[n-1] << "\n";

  cudaFree(dA); cudaFree(dB);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}