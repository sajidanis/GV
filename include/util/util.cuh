#ifndef UTIL_CUH
#define UTIL_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<typename T>
void printHostVector(const std::string name, const thrust::host_vector<T> &vec);

template<typename T>
void printDeviceVector(const std::string name, const thrust::device_vector<T>& d_vec);

unsigned long computeNearestPowerOf2(unsigned long num);

#endif
