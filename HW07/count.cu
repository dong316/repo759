#include "count.cuh"

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts)
{
    // copy input because we must sort
    thrust::device_vector<int> temp = d_in;

    // 1️⃣ sort
    thrust::sort(temp.begin(), temp.end());

    int n = temp.size();

    // worst case: all unique → allocate n
    values.resize(n);
    counts.resize(n);

    // 2️⃣ reduce_by_key
    auto new_end = thrust::reduce_by_key(
        temp.begin(),
        temp.end(),
        thrust::constant_iterator<int>(1),   // each element contributes 1
        values.begin(),
        counts.begin()
    );

    // number of unique keys
    int new_size = new_end.first - values.begin();

    values.resize(new_size);
    counts.resize(new_size);
}