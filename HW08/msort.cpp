#include "msort.h"
#include <vector>
#include <algorithm>

void merge(int* arr, std::size_t left, std::size_t mid, std::size_t right)
{
    std::vector<int> temp(right - left);

    std::size_t i = left;
    std::size_t j = mid;
    std::size_t k = 0;

    while (i < mid && j < right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < mid) temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];

    std::copy(temp.begin(), temp.end(), arr + left);
}

void msort_rec(int* arr,
               std::size_t left,
               std::size_t right,
               std::size_t threshold)
{
    std::size_t n = right - left;

    if (n <= threshold)
    {
        std::sort(arr + left, arr + right);
        return;
    }

    std::size_t mid = left + n / 2;

    #pragma omp task shared(arr)
    msort_rec(arr, left, mid, threshold);

    #pragma omp task shared(arr)
    msort_rec(arr, mid, right, threshold);

    #pragma omp taskwait
    merge(arr, left, mid, right);
}

void msort(int* arr,
           const std::size_t n,
           const std::size_t threshold)
{
    #pragma omp parallel
    {
        #pragma omp single
        msort_rec(arr, 0, n, threshold);
    }
}