#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include "matmul.h"

int main(int argc, char** argv)
{
    if (argc != 3) return 0;

    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);

    omp_set_num_threads(t);

    std::vector<float> A(n*n, 1.0f);
    std::vector<float> B(n*n, 1.0f);
    std::vector<float> C(n*n, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    mmul(A.data(), B.data(), C.data(), n);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << C[0] << std::endl;
    std::cout << C[n*n-1] << std::endl;
    std::cout << time_ms << std::endl;

    return 0;
}