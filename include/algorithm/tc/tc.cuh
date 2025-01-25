#ifndef TC_CUH
#define TC_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"
#include "tc_kernel.cuh"

void static_tc(GraphVine *graph, thrust::device_vector<float> &d_triangleCount);

#endif