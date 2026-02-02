#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) {
    long k = (long)(m - 1) / 2;

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float s = 0.0f;

            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    long ii = (long)x + (long)i - k;
                    long jj = (long)y + (long)j - k;

                    bool in_i = (0 <= ii && ii < (long)n);
                    bool in_j = (0 <= jj && jj < (long)n);

                    float val;
                    if (in_i && in_j)            val = image[ii * (long)n + jj];
                    else if (in_i || in_j)       val = 1.0f;   // edge (not corner)
                    else                         val = 0.0f;   // corner

                    s += mask[i * m + j] * val;
                }
            }

            output[x * n + y] = s;
        }
    }
}
