#include "cluster.h"

#include <cmath>
#include <cstddef>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
  // 64-byte cache line / 4-byte float = 16 floats per cache line
  constexpr size_t PAD = 16;

#pragma omp parallel num_threads(t)
  {
    const size_t tid = static_cast<size_t>(omp_get_thread_num());

    // Each thread works on one contiguous partition
    const size_t chunk = n / t;
    const size_t start = tid * chunk;
    const size_t end = start + chunk;

    const size_t padded_idx = tid * PAD;

    float local_sum = 0.0f;
    const float center = centers[padded_idx];

    for (size_t i = start; i < end; ++i) {
      local_sum += std::fabs(arr[i] - center);
    }

    // One write per thread, separated by padding to avoid false sharing
    dists[padded_idx] = local_sum;
  }
}
