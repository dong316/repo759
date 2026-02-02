#include <iostream>
#include <random>
#include <chrono>
#include "scan.h"

int main(int argc, char** argv) {
    // 1) 从命令行读 n
    std::size_t n = std::stoull(argv[1]);

    // 2) 分配数组
    float* arr = new float[n];
    float* out = new float[n];

    // 3) 生成 [-1, 1] 的随机数
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(gen);
    }

    // 4) 计时 + scan
    auto t0 = std::chrono::high_resolution_clock::now();
    scan(arr, out, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 5) 输出（严格按题目顺序）
    std::cout << ms << "\n";
    std::cout << out[0] << "\n";
    std::cout << out[n - 1] << "\n";

    // 6) 释放内存
    delete[] arr;
    delete[] out;
}
