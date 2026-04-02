#include <iostream>
#include <chrono>
#include <cstdlib>
#include "optimize.h"

using namespace std;
using namespace std::chrono;

double run_avg(void (*func)(vec *, data_t *), vec *v, data_t &dest, int repeat = 10) {
    double total_ms = 0.0;

    for (int r = 0; r < repeat; r++) {
        auto start = high_resolution_clock::now();
        func(v, &dest);
        auto end = high_resolution_clock::now();
        total_ms += duration<double, milli>(end - start).count();
    }

    return total_ms / repeat;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: ./task1 n\n";
        return 1;
    }

    size_t n = static_cast<size_t>(atoll(argv[1]));
    if (n == 0) {
        cerr << "n must be positive\n";
        return 1;
    }

    vec v(n);
    v.data = new data_t[n];

    // safe values for both + and *
    for (size_t i = 0; i < n; i++) {
        v.data[i] = 1;
    }

    data_t dest = IDENT;
    double t1 = run_avg(optimize1, &v, dest);
    cout << "optimize1_result " << dest << "\n";
    cout << "optimize1_time_ms " << t1 << "\n";

    dest = IDENT;
    double t2 = run_avg(optimize2, &v, dest);
    cout << "optimize2_result " << dest << "\n";
    cout << "optimize2_time_ms " << t2 << "\n";

    dest = IDENT;
    double t3 = run_avg(optimize3, &v, dest);
    cout << "optimize3_result " << dest << "\n";
    cout << "optimize3_time_ms " << t3 << "\n";

    dest = IDENT;
    double t4 = run_avg(optimize4, &v, dest);
    cout << "optimize4_result " << dest << "\n";
    cout << "optimize4_time_ms " << t4 << "\n";

    dest = IDENT;
    double t5 = run_avg(optimize5, &v, dest);
    cout << "optimize5_result " << dest << "\n";
    cout << "optimize5_time_ms " << t5 << "\n";

    delete[] v.data;
    return 0;
}