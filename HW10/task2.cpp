#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include "reduce.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "This program must be run with exactly 2 processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: ./task2 n t\n";
        }
        MPI_Finalize();
        return 1;
    }

    size_t n = static_cast<size_t>(atoll(argv[1]));
    int t = atoi(argv[2]);

    omp_set_num_threads(t);

    // each MPI process creates its own local array of length n
    std::vector<float> arr(n);

    std::mt19937 gen(1234 + rank);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        arr[i] = dist(gen);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    float res = reduce(arr.data(), 0, n);

    float global_res = 0.0f;
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << global_res << "\n";
        std::cout << (end - start) * 1000.0 << "\n";
    }

    MPI_Finalize();
    return 0;
}