#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include "matrix_utils.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./jacobi_gs N [max_iter] [tol] [seed]" << endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int max_iter = 10000;
    double tol = 1e-6;
    unsigned int seed = 42;

    if (argc >= 3) max_iter = atoi(argv[2]);
    if (argc >= 4) tol = atof(argv[3]);
    if (argc >= 5) seed = (unsigned int) atoi(argv[4]);

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);
    vector<double> x(N, 0.0);

    generate_diagonally_dominant_system(A, b, N, seed);

    // warm-up run (not timed)
    vector<double> x_warmup(N, 0.0);
    for (int i = 0; i < N; i++) {
        double sigma = 0.0;
        for (int j = 0; j < N; j++) {
            if (j != i) {
                sigma += A[i][j] * x_warmup[j];
            }
        }
        x_warmup[i] = (b[i] - sigma) / A[i][i];
    }

    auto start = chrono::high_resolution_clock::now();

    int iter;
    double err = 0.0;

    for (iter = 0; iter < max_iter; iter++) {
        err = 0.0;

        for (int i = 0; i < N; i++) {
            double old_x = x[i];
            double sigma = 0.0;

            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sigma += A[i][j] * x[j];
                }
            }

            x[i] = (b[i] - sigma) / A[i][i];
            err += fabs(x[i] - old_x);
        }

        if (err < tol) {
            break;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double runtime = chrono::duration<double>(end - start).count();

    double residual = 0.0;
    for (int i = 0; i < N; i++) {
        double ax_i = 0.0;
        for (int j = 0; j < N; j++) {
            ax_i += A[i][j] * x[j];
        }
        residual += fabs(ax_i - b[i]);
    }

    cout << "Residual L1 = " << residual << endl;

    cout << "Gauss-Seidel serial solver finished" << endl;
    cout << "N = " << N << endl;
    cout << "Max iterations = " << max_iter << endl;
    cout << "Tolerance = " << tol << endl;
    cout << "Seed = " << seed << endl;
    cout << "Iterations used = " << iter + 1 << endl;
    cout << "Final error = " << err << endl;
    cout << "Runtime (s) = " << runtime << endl;

    ofstream file("results/runtime_gs.csv", ios::app);
    file << "gs_serial," << N << "," << iter + 1 << "," << runtime << endl;
    file.close();

    return 0;
}