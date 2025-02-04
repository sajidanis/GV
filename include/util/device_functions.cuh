#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

#include "graphvine.cuh"

// Device Functions
    
extern __device__ unsigned long bit_string_lookup[BIT_STRING_LOOKUP_SIZE];

__device__ unsigned long device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size);

__device__ EdgeBlock *pop_edge_block_address(unsigned long pop_count, unsigned long *d_prefix_sum_edge_blocks, unsigned long current_vertex);

__device__ void insert_edge_block_to_CBT_v2(EdgeBlock *root, unsigned long bit_string, unsigned long length, EdgeBlock *new_block, EdgeBlock *last_insert_edge_block, unsigned long batch_index_counter, unsigned long global_index_counter, unsigned long current_edge_block_count, unsigned long total_edge_blocks_new, unsigned long source_vertex, unsigned long id);

__device__ EdgeBlock *traverse_bit_string(EdgeBlock *root, unsigned long bit_string);

__device__ unsigned long tc_device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size);

__device__ unsigned long long tc_final_device_binary_search(unsigned long *input_array, unsigned long long key, unsigned long long size);

__device__ unsigned long traversal_string(unsigned long val, unsigned long *length);

#endif