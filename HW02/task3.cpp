#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "matmul.h"

static double run_and_report(void (*mm)(const double*, const double*, double*, unsigned int),
                             const double* A, const double* B, double* C, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) C[i] = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    mm(A, B, C, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    const unsigned int n = 1024; // >= 1000，且是 2 的幂，常用 benchmark size

    std::vector<double> A(n * n), B(n * n);
    std::vector<double> Avec = A, Bvec = B; // 先占位，下面会真正填
    double* C = new double[n * n];

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    Avec = A;
    Bvec = B;

    std::cout << n << "\n";

    double t;

    t = run_and_report(mmul1, A.data(), B.data(), C, n);
    std::cout << t << "\n" << C[n * n - 1] << "\n";

    t = run_and_report(mmul2, A.data(), B.data(), C, n);
    std::cout << t << "\n" << C[n * n - 1] << "\n";

    t = run_and_report(mmul3, A.data(), B.data(), C, n);
    std::cout << t << "\n" << C[n * n - 1] << "\n";

    for (unsigned int i = 0; i < n * n; ++i) C[i] = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    mmul4(Avec, Bvec, C, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double, std::milli>(t1 - t0).count() << "\n";
    std::cout << C[n * n - 1] << "\n";

    delete[] C;
}
