#ifndef BC_KERNEL_CUH
#define BC_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"

__global__ void initializeArrays(int n, int *sigma, float *delta, int *dist);

__global__ void initbfsKernel(int source, int *sigma, int *dist, int *queue, int *queue_size);

__global__ void bfsKernel(VertexDictionary *d_vertex_dict, int *sigma, int *dist, int *frontier, int *frontier_size, int *next_frontier, int *next_frontier_size);

__global__ void bfsKernelSimplified(VertexDictionary *d_vertex_dict, int *sigma, int *dist, int *frontier, int *frontier_size, int *next_frontier, int *next_frontier_size);

__global__ void dependencyKernel(VertexDictionary *d_vertex_dict, float *delta, int *sigma, int *dist, float *bc, int vertex_size, int source_vertex);

#endif