#ifndef TC_KERNEL_CUH
#define TC_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"

__global__ void tc_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangle_count);

__global__ void dynamic_tc_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangle_count, unsigned long *d_affected_nodes);

#endif