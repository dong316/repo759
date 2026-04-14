#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>

void generate_diagonally_dominant_system(
    std::vector<std::vector<double>> &A,
    std::vector<double> &b,
    int N
);

#endif