#include "matrix_utils.h"
#include <cstdlib>
#include <cmath>

using namespace std;

void generate_diagonally_dominant_system(
    vector<vector<double>> &A,
    vector<double> &b,
    int N,
    unsigned int seed
) {
    srand(seed);

    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;

        for (int j = 0; j < N; j++) {
            if (i != j) {
                A[i][j] = (double) rand() / RAND_MAX;
                row_sum += fabs(A[i][j]);
            } else {
                A[i][j] = 0.0;
            }
        }

        A[i][i] = row_sum + 1.0;
        b[i] = (double) rand() / RAND_MAX;
    }
}


void generate_stencil_system(
    vector<vector<double>> &A,
    vector<double> &b,
    int N
) {
    for (int i = 0; i < N; i++) {
        // initialize row to zero
        for (int j = 0; j < N; j++) {
            A[i][j] = 0.0;
        }

        // 1D stencil-like tridiagonal system
        if (i > 0) {
            A[i][i - 1] = -1.0;
        }

        if (i < N - 1) {
            A[i][i + 1] = -1.0;
        }

        A[i][i] = 2.0;
        b[i] = 1.0;
    }
}
