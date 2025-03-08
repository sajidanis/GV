#ifndef PR_KERNEL_CUH
#define PR_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaError.cuh"

#include "graphvine.cuh"

__global__ void pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_pageRankVector_1, float *d_pageRankVector_2);

__global__ void pagerank_post_kernel(VertexDictionary *device_vertex_dictionary, size_t vertex_size, float damp, float normalized_damp, float *d_pageRankVector_1, float *d_pageRankVector_2, float *reduction);

__global__ void dynamic_pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_affected_nodes, float *d_pageRankVector_1_pointer, float *d_pageRankVector_2_pointer);

__global__ void pagerank_post_kernel_dynamic(VertexDictionary *device_vertex_dictionary, size_t vertex_size, float damp, float normalized_damp, float *d_prev_pr, float *d_curr_pr, float *reduction, unsigned long *d_affected_nodes);

#endif