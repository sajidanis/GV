#ifndef DELETE_KERNELS
#define DELETE_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"
#include "device_functions.cuh"

__global__ void batch_delete_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees);

__global__ void compactionVertexCentric(unsigned long totalvertices, VertexDictionary *device_vertex_dictionary);

#endif