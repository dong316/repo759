#include <iostream>
#include <omp.h>

long factorial(int n)
{
    long f = 1;
    for (int i = 1; i <= n; i++)
        f *= i;
    return f;
}

int main()
{
    omp_set_num_threads(4);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // ⭐ 只打印一次线程总数
#pragma omp single
        {
            std::cout << "Number of threads: " << nthreads << std::endl;
        }

        // ⭐ 每个线程自我介绍
        std::cout << "I am thread No. " << tid << std::endl;

        // ⭐ 并行计算 factorial
#pragma omp for
        for (int i = 1; i <= 8; i++)
        {
            long f = factorial(i);

#pragma omp critical
            {
                std::cout << i << "!=" << f << std::endl;
            }
        }
    }

    return 0;
}