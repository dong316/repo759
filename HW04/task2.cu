// task2.cu
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

#define CUDA_OK(x) do { cudaError_t e=(x); if(e){ \
  fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(e)); exit(1);} } while(0)

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "Usage: %s n R threads_per_block\n", argv[0]); return 1; }
  unsigned int n   = (unsigned int)std::strtoul(argv[1], nullptr, 10);
  unsigned int R   = (unsigned int)std::strtoul(argv[2], nullptr, 10);
  unsigned int tpb = (unsigned int)std::strtoul(argv[3], nullptr, 10);

  float *hImg = new float[n], *hMask = new float[2*R + 1];

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (unsigned int i=0;i<n;i++) hImg[i]=dist(rng);
  for (unsigned int i=0;i<2*R+1;i++) hMask[i]=dist(rng);

  float *dImg, *dMask, *dOut;
  CUDA_OK(cudaMalloc(&dImg,  n * sizeof(float)));
  CUDA_OK(cudaMalloc(&dOut,  n * sizeof(float)));
  CUDA_OK(cudaMalloc(&dMask, (2*R + 1) * sizeof(float)));
  CUDA_OK(cudaMemcpy(dImg,  hImg,  n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dMask, hMask, (2*R + 1) * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t st, ed; CUDA_OK(cudaEventCreate(&st)); CUDA_OK(cudaEventCreate(&ed));
  CUDA_OK(cudaEventRecord(st));
  stencil(dImg, dMask, dOut, n, R, tpb);
  CUDA_OK(cudaEventRecord(ed));
  CUDA_OK(cudaEventSynchronize(ed));

  float ms=0.0f; CUDA_OK(cudaEventElapsedTime(&ms, st, ed));
  float last=0.0f; CUDA_OK(cudaMemcpy(&last, dOut + (n-1), sizeof(float), cudaMemcpyDeviceToHost));

  printf("%.2f\n%.2f\n", last, ms);

  CUDA_OK(cudaEventDestroy(st)); CUDA_OK(cudaEventDestroy(ed));
  CUDA_OK(cudaFree(dImg)); CUDA_OK(cudaFree(dMask)); CUDA_OK(cudaFree(dOut));
  delete[] hImg; delete[] hMask;
  return 0;
}