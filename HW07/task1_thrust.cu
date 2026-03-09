#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./task1_thrust n\n";
        return 1;
    }

    int n = atoi(argv[1]);

    // host vector
    thrust::host_vector<float> h_vec(n);

    // fill with random numbers [-1,1]
    for (int i = 0; i < n; i++) {
        h_vec[i] = -1.0f + 2.0f * rand() / RAND_MAX;
    }

    // copy to device
    thrust::device_vector<float> d_vec = h_vec;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // reduction
    float result = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << result << std::endl;
    std::cout << ms << std::endl;

    return 0;
}