#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include "convolution.h"

int main(int argc, char** argv)
{
    if (argc != 3) return 0;

    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);

    omp_set_num_threads(t);

    std::size_t m = 3;   // mask size fixed

    std::vector<float> image(n*n, 1.0f);
    std::vector<float> output(n*n, 0.0f);
    std::vector<float> mask(m*m, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    convolve(image.data(), output.data(), n, mask.data(), m);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << output[0] << std::endl;
    std::cout << output[n*n - 1] << std::endl;
    std::cout << time_ms << std::endl;

    return 0;
}