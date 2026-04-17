#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include "matrix_utils.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./jacobi_omp N [max_iter] [tol] [seed]" << endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int max_iter = 10000;
    double tol = 1e-6;
    unsigned int seed = 42;

    if (argc >= 3) {
        max_iter = atoi(argv[2]);
    }
    if (argc >= 4) {
        tol = atof(argv[3]);
    }
    if (argc >= 5) {
        seed = (unsigned int) atoi(argv[4]);
    }

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);
    vector<double> x(N, 0.0);
    vector<double> x_new(N, 0.0);

    generate_diagonally_dominant_system(A, b, N, seed);

    auto start = chrono::high_resolution_clock::now();

    int iter;
    double err = 0.0;

    for (iter = 0; iter < max_iter; iter++) {

        // parallelize outer loop
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            double sigma = 0.0;

            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sigma += A[i][j] * x[j];
                }
            }

            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        err = 0.0;

        // reduction for error
        #pragma omp parallel for reduction(+:err)
        for (int i = 0; i < N; i++) {
            err += fabs(x_new[i] - x[i]);
        }

        // update x
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] = x_new[i];
        }

        if (err < tol) {
            break;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end - start).count();

    cout << "Jacobi OpenMP solver finished" << endl;
    cout << "N = " << N << endl;
    cout << "Max iterations = " << max_iter << endl;
    cout << "Tolerance = " << tol << endl;
    cout << "Seed = " << seed << endl;
    cout << "Threads = " << omp_get_max_threads() << endl;
    cout << "Iterations used = " << iter + 1 << endl;
    cout << "Final error = " << err << endl;
    cout << "Runtime (s) = " << runtime << endl;

    ofstream file("results/runtime.csv", ios::app);
    file << "omp," << N << "," << iter + 1 << "," << runtime << endl;
    file.close();

    return 0;
}