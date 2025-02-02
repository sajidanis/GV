#ifndef TC_KERNEL_CUH
#define TC_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"

__global__ void triangle_counting_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_csr_offset, unsigned long *d_csr_edges);

__global__ void dynamic_triangle_counting_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_affected_nodes);

__global__ void tc_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount);

#endif