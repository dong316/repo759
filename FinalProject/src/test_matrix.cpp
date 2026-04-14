#include <iostream>
#include <vector>
#include <cmath>
#include "matrix_utils.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./test_matrix N" << endl;
        return 1;
    }

    int N = atoi(argv[1]);

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);

    generate_diagonally_dominant_system(A, b, N);

    cout << "Generated a " << N << " x " << N << " matrix." << endl;

    int rows_to_print = (N < 5) ? N : 5;

    cout << "\nFirst " << rows_to_print << " rows of A:" << endl;
    for (int i = 0; i < rows_to_print; i++) {
        for (int j = 0; j < rows_to_print; j++) {
            cout << A[i][j] << "\t";
        }
        cout << endl;
    }

    cout << "\nFirst " << rows_to_print << " entries of b:" << endl;
    for (int i = 0; i < rows_to_print; i++) {
        cout << b[i] << endl;
    }

    cout << "\nChecking diagonal dominance for first " << rows_to_print << " rows:" << endl;
    for (int i = 0; i < rows_to_print; i++) {
        double off_diag_sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                off_diag_sum += fabs(A[i][j]);
            }
        }

        cout << "Row " << i
             << ": |A[i][i]| = " << fabs(A[i][i])
             << ", sum(off-diagonal) = " << off_diag_sum;

        if (fabs(A[i][i]) > off_diag_sum) {
            cout << "  --> OK";
        } else {
            cout << "  --> NOT OK";
        }
        cout << endl;
    }

    return 0;
}