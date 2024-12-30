#ifndef UTIL_CUH
#define UTIL_CUH

#include <thrust/host_vector.h>

template<typename T>
void printHostVector(const std::string name, const thrust::host_vector<T> &vec);

unsigned long computeNearestPowerOf2(unsigned long num);

#endif
