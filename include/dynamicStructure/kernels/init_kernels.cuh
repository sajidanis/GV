#ifndef INIT_KERNELS
#define INIT_KERNELS

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphStructures.cuh"
#include "graphvine.cuh"

__global__ void data_structure_init(VertexDictionary *device_vertex_dictionary, unsigned long *d_vertex_id, unsigned long *d_edge_block_count, unsigned int *d_active_edge_count, unsigned long *d_last_insert_edge_offset, EdgeBlock **d_last_insert_edge_block, EdgeBlock **d_edge_block_address, EdgeBlock **d_queue_edge_block_address);

__global__ void parallel_push_edge_preallocate_list_to_device_queue(EdgeBlock *d_edge_preallocate_list, unsigned long total_edge_blocks_count_init, unsigned long device_edge_block_capacity);

__global__ void parallel_push_queue_update(unsigned long total_edge_blocks_count_init);

__global__ void parallel_vertex_dictionary_init_v1(unsigned long vertex_size, VertexDictionary *device_vertex_dictionary);

__global__ void printVertexDictionaryKernel(const VertexDictionary* d_vertex_dict, size_t vertex_size);

__global__ void printEdgeBlockQueue(EdgeBlock **d_queue_edge_block_address, size_t queue_size);

#endif