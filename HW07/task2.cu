#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>

#include "count.cuh"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "usage: ./task2 n\n";
        return 0;
    }

    int n = atoi(argv[1]);

    // 1️⃣ host random data
    thrust::host_vector<int> h_vec(n);
    for (int i = 0; i < n; i++)
        h_vec[i] = rand() % 501;

    // 2️⃣ copy to device
    thrust::device_vector<int> d_in = h_vec;

    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    count(d_in, values, counts);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 3️⃣ print required outputs
    std::cout << values.back() << std::endl;
    std::cout << counts.back() << std::endl;
    std::cout << ms << std::endl;

    return 0;
}