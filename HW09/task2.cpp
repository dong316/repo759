#include "montecarlo.h"

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
  const int t = std::stoi(argv[2]);

  if (n == 0 || t < 1 || t > 10) {
    return 1;
  }

  omp_set_num_threads(t);

  const float radius = 1.0f;

  std::vector<float> x(n);
  std::vector<float> y(n);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-radius, radius);

  for (size_t i = 0; i < n; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }

  const auto start = std::chrono::high_resolution_clock::now();
  const int incircle = montecarlo(n, x.data(), y.data(), radius);
  const auto end = std::chrono::high_resolution_clock::now();

  const double pi_est = 4.0 * static_cast<double>(incircle) / static_cast<double>(n);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << std::fixed << std::setprecision(4) << pi_est << '\n';
  std::cout << std::fixed << std::setprecision(3) << elapsed_ms << '\n';

  return 0;
}