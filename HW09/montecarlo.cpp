#include "montecarlo.h"

#include <cstddef>

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
  int incircle = 0;
  const float r2 = radius * radius;

#ifdef USE_SIMD
#pragma omp parallel for simd reduction(+ : incircle)
#else
#pragma omp parallel for reduction(+ : incircle)
#endif
  for (size_t i = 0; i < n; ++i) {
    const float d2 = x[i] * x[i] + y[i] * y[i];
    if (d2 <= r2) {
      incircle += 1;
    }
  }

  return incircle;
}
