#ifndef INSERT_KERNELS
#define INSERT_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"
#include "device_functions.cuh"

__global__ void batched_edge_inserts_EC(EdgeBlock *d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, VertexDictionary *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees);

__global__ void batched_edge_inserts_EC_postprocessing(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees, unsigned long *d_prefix_sum_edge_blocks, unsigned long batch_number);

__global__ void update_edge_queue(unsigned long pop_count);

#endif