#include "cluster.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    return 1;
  }

  const size_t n = static_cast<size_t>(std::stoull(argv[1]));
  const size_t t = static_cast<size_t>(std::stoull(argv[2]));

  if (n == 0 || t == 0 || t > 10) {
    return 1;
  }

  // Padding so each thread's entry is on a separate cache line
  constexpr size_t PAD = 16;

  // 1) Create and fill arr with random floats in [0, n]
  std::vector<float> arr(n);
  std::mt19937 rng(42);  // fixed seed for reproducibility
  std::uniform_real_distribution<float> dist_rand(0.0f, static_cast<float>(n));

  for (size_t i = 0; i < n; ++i) {
    arr[i] = dist_rand(rng);
  }

  // 2) Sort arr
  std::sort(arr.begin(), arr.end());

  // 3) Create and fill centers
  // centers should be [ n/(2t), 3n/(2t), ..., (2t-1)n/(2t) ]
  std::vector<float> centers(t * PAD, 0.0f);
  for (size_t i = 0; i < t; ++i) {
    centers[i * PAD] =
        static_cast<float>((2.0 * static_cast<double>(i) + 1.0) *
                           static_cast<double>(n) / (2.0 * static_cast<double>(t)));
  }

  // 4) Create dists and initialize to zero
  std::vector<float> dists(t * PAD, 0.0f);

  // 5) Call cluster and time only cluster
  const auto start = std::chrono::high_resolution_clock::now();
  cluster(n, t, arr.data(), centers.data(), dists.data());
  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // 6) Calculate maximum distance in dists
  float max_dist = dists[0];
  size_t max_id = 0;

  for (size_t i = 1; i < t; ++i) {
    const float val = dists[i * PAD];
    if (val > max_dist) {
      max_dist = val;
      max_id = i;
    }
  }

  // 7) Print outputs
  std::cout << max_dist << '\n';
  std::cout << max_id << '\n';
  std::cout << std::fixed << std::setprecision(3) << elapsed_ms << '\n';

  return 0;
}