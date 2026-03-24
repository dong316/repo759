#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
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

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: ./task3 n\n";
        }
        MPI_Finalize();
        return 1;
    }

    long long n = std::atoll(argv[1]);
    if (n <= 0) {
        if (rank == 0) {
            std::cerr << "n must be a positive integer.\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<float> sendbuf(n), recvbuf(n);

    for (long long i = 0; i < n; i++) {
        sendbuf[i] = static_cast<float>(rank + 0.001 * i);
        recvbuf[i] = 0.0f;
    }

    int peer = 1 - rank;
    int tag  = 0;

    MPI_Status status;
    double t_local = 0.0;

    if (rank == 0) {
        double start = MPI_Wtime();

        MPI_Send(sendbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &status);

        double end = MPI_Wtime();
        t_local = (end - start) * 1000.0;   // ms

        double t1 = 0.0;
        MPI_Recv(&t1, 1, MPI_DOUBLE, peer, 1, MPI_COMM_WORLD, &status);

        std::cout << (t_local + t1) << std::endl;
    } else {
        double start = MPI_Wtime();

        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &status);
        MPI_Send(sendbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD);

        double end = MPI_Wtime();
        t_local = (end - start) * 1000.0;   // ms

        MPI_Send(&t_local, 1, MPI_DOUBLE, peer, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
