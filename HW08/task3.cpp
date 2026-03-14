#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include "msort.h"

int main(int argc, char** argv)
{
    if (argc != 4) return 0;

    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);
    std::size_t ts = std::stoul(argv[3]);

    omp_set_num_threads(t);

    std::vector<int> arr(n);

    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (std::size_t i = 0; i < n; ++i)
        arr[i] = dist(gen);

    auto start = std::chrono::high_resolution_clock::now();

    msort(arr.data(), n, ts);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << arr[0] << std::endl;
    std::cout << arr[n-1] << std::endl;
    std::cout << time_ms << std::endl;

    return 0;
}