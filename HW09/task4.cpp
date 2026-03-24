#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <omp.h>
#include "convolve.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: ./task4 n\n";
        return 1;
    }

    std::size_t n = std::atoll(argv[1]);
    std::size_t m = 3;

    float* image  = new float[n * n];
    float* output = new float[n * n];
    float* mask   = new float[m * m];

    for (std::size_t i = 0; i < n * n; i++) {
        image[i] = -10.0f + 20.0f * (float)rand() / (float)RAND_MAX;
    }

    for (std::size_t i = 0; i < m * m; i++) {
        mask[i] = -1.0f + 2.0f * (float)rand() / (float)RAND_MAX;
    }

    double t0 = omp_get_wtime();
    convolve(image, output, n, mask, m);
    double t1 = omp_get_wtime();

    std::cout << (t1 - t0) * 1000.0 << std::endl;

    delete[] image;
    delete[] output;
    delete[] mask;

    return 0;
}