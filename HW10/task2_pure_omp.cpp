#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <omp.h>
#include "reduce.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2_pure_omp n t\n";
        return 1;
    }

    size_t n = static_cast<size_t>(atoll(argv[1]));
    int t = atoi(argv[2]);

    omp_set_num_threads(t);

    std::vector<float> arr(n);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }

    double start = omp_get_wtime();
    float res = reduce(arr.data(), 0, n);
    double end = omp_get_wtime();

    std::cout << res << "\n";
    std::cout << (end - start) * 1000.0 << "\n";

    return 0;
}