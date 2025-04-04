#ifndef INSERT_KERNELS
#define INSERT_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "graphvine.cuh"
#include "device_functions.cuh"

__global__ void batched_edge_inserts_EC(EdgeBlock *d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, VertexDictionary *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees);

__global__ void batched_edge_inserts_EC_postprocessing(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees, unsigned long *d_prefix_sum_edge_blocks, unsigned long batch_number);

__global__ void update_edge_queue(unsigned long pop_count);

__global__ void device_remove_batch_duplicates(unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees);

__global__ void device_update_source_degrees(unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees);

__global__ void batched_delete_preprocessing_EC_LD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector);

__global__ void batched_delete_kernel_EC_LD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector);

__global__ void batched_delete_kernel_EC_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector);

__global__ void device_insert_preprocessing(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long batch_size, unsigned long *d_source_degrees, unsigned long batch_number, unsigned long *d_edge_blocks_count);

__global__ void find_affected_nodes(unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_affected_nodes);

__global__ void device_sorting_post();

__global__ void cub_sort_edge_blocks();

#endif