#ifndef PR_KERNEL_CUH
#define PR_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"

__global__ void pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_pageRankVector_1, float *d_pageRankVector_2, unsigned long *d_source_vector);

__global__ void dynamic_pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_affected_nodes, float *d_pageRankVector_1_pointer, float *d_pageRankVector_2_pointer);

#endif