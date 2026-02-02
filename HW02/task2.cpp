#include <iostream>
#include <random>
#include <chrono>
#include "convolution.h"

int main(int argc, char** argv) {
    std::size_t n = std::stoull(argv[1]);
    std::size_t m = std::stoull(argv[2]);

    float* image = new float[n * n];
    float* mask  = new float[m * m];
    float* output = new float[n * n];

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) image[i] = dist_img(gen);
    for (std::size_t i = 0; i < m * m; ++i) mask[i]  = dist_mask(gen);

    auto t0 = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << ms << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";

    delete[] image;
    delete[] mask;
    delete[] output;
}
