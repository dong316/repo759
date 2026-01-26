#include <cstdio>   // for printf
#include <cstdlib>  // for atoi
#include <iostream> // for std::cout

int main(int argc, char* argv[]) {
    // Check that N is provided
    if (argc < 2) {
        return 1;
    }

    int N = std::atoi(argv[1]);

    // b) Print 0 to N using printf
    for (int i = 0; i <= N; ++i) {
        std::printf("%d", i);
        if (i < N) {
            std::printf(" ");
        }
    }
    std::printf("\n");

    // c) Print N to 0 using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i > 0) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;

    return 0;
}
