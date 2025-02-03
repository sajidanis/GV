#ifndef UTIL_KERNELS
#define UTIL_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"
#include "device_functions.cuh"

__global__ void printDeviceVertexDictionary_kernel(size_t vertexSize, VertexDictionary* vertexDictionary);


#endif