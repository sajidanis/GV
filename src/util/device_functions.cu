#include "device_functions.cuh"

__device__ unsigned long bit_string_lookup[BIT_STRING_LOOKUP_SIZE];

__device__ unsigned long device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size)
{
    long start = 0;
    long end = size;
    long mid;

    while (start <= end){
        mid = (start + end) / 2;

        // Check if x is present at mid
        if ((input_array[mid] == key) && ((mid + 1) <= end) && (input_array[mid + 1] > key))
            return mid;

        // If x is smaller, ignore right half
        if (input_array[mid] > key)
            end = mid - 1;
        // If x greater, ignore left half
        else
            start = mid + 1;
    }

    // If we reach here, then element was not present
    return start - 1;
}

__device__ EdgeBlock *pop_edge_block_address(unsigned long pop_count, unsigned long *d_prefix_sum_edge_blocks, unsigned long current_vertex)
{

    if ((d_e_queue.count < pop_count) || (d_e_queue.front == -1)) {
        return NULL;
    } else {
        return d_e_queue.edge_block_address[d_e_queue.front + d_prefix_sum_edge_blocks[current_vertex]];
    }

    // __syncthreads();
}

__device__ void insert_edge_block_to_CBT_v2(EdgeBlock *root, unsigned long bit_string, unsigned long length, EdgeBlock *new_block, EdgeBlock *last_insert_edge_block, unsigned long batch_index_counter, unsigned long global_index_counter, unsigned long current_edge_block_count, unsigned long total_edge_blocks_new, unsigned long source_vertex, unsigned long id) {
    EdgeBlock *curr = root;
    // indices from 0
    unsigned long current_capacity = 0;
    if (current_edge_block_count)
        current_capacity = (current_edge_block_count - 1) * 2 + 2;

    if ((global_index_counter <= current_capacity) && (root != NULL)) {
        for (; bit_string > 10; bit_string /= 10) {
            if (bit_string % 2)
                curr = curr->lptr;
            else
                curr = curr->rptr;
        }

        if (bit_string % 2)
            curr->lptr = new_block;
        else
            curr->rptr = new_block;
    }

    if (batch_index_counter)
        new_block->level_order_predecessor = new_block - 1;
    else
        new_block->level_order_predecessor = last_insert_edge_block;

    if (((global_index_counter * 2) + 2) < total_edge_blocks_new) {

        new_block->lptr = new_block + (global_index_counter * 2) + 1 - global_index_counter;
        new_block->rptr = new_block + (global_index_counter * 2) + 2 - global_index_counter;
    }

    else if (((global_index_counter * 2) + 1) < total_edge_blocks_new) {

        new_block->lptr = new_block + (global_index_counter * 2) + 1 - global_index_counter;
        new_block->rptr = NULL;
    }

    else {
        new_block->lptr = NULL;
        new_block->rptr = NULL;
    }
}

__device__ EdgeBlock *traverse_bit_string(EdgeBlock *root, unsigned long bit_string) {

    EdgeBlock *curr = root;

    for (; bit_string > 0; bit_string /= 10){
        if (bit_string % 2){
            curr = curr->lptr;
        }
        else{
            curr = curr->rptr;
        }
    }

    return curr;
}

__device__ unsigned long tc_device_binary_search(unsigned long *input_array, unsigned long key, unsigned long size) {

    long start = 0;
    long end = (long)size;
    long mid;

    while (start <= end) {

        mid = (start + end) / 2;

        unsigned long item = input_array[mid] - 1;
        if (item == key) {
            return mid + 1;
        }

        if (item < key)
            start = mid + 1;
        else
            end = mid - 1;
    }
    return start;
}

__device__ unsigned long long tc_final_device_binary_search(unsigned long *input_array, unsigned long long key, unsigned long long size) {
    unsigned long start = 0;
    unsigned long end = size;
    unsigned long mid;

    while (start <= end) {

        mid = (start + end) / 2;

        unsigned long item = input_array[mid] - 1;

        // Check if x is present at mid
        if (item == key)
            return 1;

        // If x greater, ignore left half
        if (item < key)
            start = mid + 1;

        // If x is smaller, ignore right half
        else
            end = mid - 1;
    }

    // If we reach here, then element was not present
    return 0;
}

__device__ unsigned long traversal_string(unsigned long val, unsigned long *length) {
    unsigned long temp = val;
    unsigned long bit_string = 0;
    *length = 0;

    while (temp > 1){
        // bit_string = ((temp % 2) * pow(10, iteration++)) + bit_string;
        if (temp % 2)
            bit_string = (bit_string * 10) + 2;
        else
            bit_string = (bit_string * 10) + 1;

        // bit_string = (bit_string * 10) + (temp % 2);
        temp = temp / 2;
    }
    return bit_string;
}
