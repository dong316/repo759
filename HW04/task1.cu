// task1.cu
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

#define CUDA_OK(x) do { cudaError_t e=(x); if(e) { \
  fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "Usage: %s n threads_per_block\n", argv[0]); return 1; }
  size_t n = (size_t)std::strtoull(argv[1], nullptr, 10);
  unsigned int tpb = (unsigned int)std::strtoul(argv[2], nullptr, 10);
  size_t N = n * n;

  float *hA = new float[N], *hB = new float[N];
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < N; i++) { hA[i] = dist(rng); hB[i] = dist(rng); }

  float *dA, *dB, *dC;
  CUDA_OK(cudaMalloc(&dA, N * sizeof(float)));
  CUDA_OK(cudaMalloc(&dB, N * sizeof(float)));
  CUDA_OK(cudaMalloc(&dC, N * sizeof(float)));
  CUDA_OK(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t st, ed; CUDA_OK(cudaEventCreate(&st)); CUDA_OK(cudaEventCreate(&ed));
  CUDA_OK(cudaEventRecord(st));
  matmul(dA, dB, dC, n, tpb);
  CUDA_OK(cudaEventRecord(ed));
  CUDA_OK(cudaEventSynchronize(ed));

  float ms = 0.0f; CUDA_OK(cudaEventElapsedTime(&ms, st, ed));
  float last = 0.0f;
  CUDA_OK(cudaMemcpy(&last, dC + (N - 1), sizeof(float), cudaMemcpyDeviceToHost));

  printf("%.2f\n%.2f\n", last, ms);

  CUDA_OK(cudaEventDestroy(st)); CUDA_OK(cudaEventDestroy(ed));
  CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dB)); CUDA_OK(cudaFree(dC));
  delete[] hA; delete[] hB;
  return 0;
}