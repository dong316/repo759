#include "matmul.h"

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
            double s = 0.0;
            for (unsigned int k = 0; k < n; ++k)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int k = 0; k < n; ++k) {
            double a = A[i * n + k];
            for (unsigned int j = 0; j < n; ++j)
                C[i * n + j] += a * B[k * n + j];
        }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    for (unsigned int j = 0; j < n; ++j)
        for (unsigned int k = 0; k < n; ++k) {
            double b = B[k * n + j];
            for (unsigned int i = 0; i < n; ++i)
                C[i * n + j] += A[i * n + k] * b;
        }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n) {
    for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j) {
            double s = 0.0;
            for (unsigned int k = 0; k < n; ++k)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}
