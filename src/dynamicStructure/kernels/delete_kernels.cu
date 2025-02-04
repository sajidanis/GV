#include "delete_kernels.cuh"

__global__ void batch_delete_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= vertex_size) return;

    unsigned long src = device_vertex_dictionary->vertex_id[id];

    unsigned long edge_blocks_count = device_vertex_dictionary->edge_block_count[id];
    unsigned long edge_count = device_vertex_dictionary->active_edge_count[id];

    if(edge_count == 0) return;

    unsigned long csr_start = d_csr_offset[id];
    unsigned long csr_end = d_csr_offset[id + 1];

    bool found = false;

    EdgeBlock *curr;

    for(size_t idx = csr_start; idx < csr_end; idx++) {

        EdgeBlock *root = device_vertex_dictionary->edge_block_address[id];
        

        unsigned long destination_vertex = d_csr_edges[idx];
        
        if(destination_vertex == INFTY) continue;

        found = false;

        for(unsigned long i = 0; i < edge_blocks_count; i++) {
            unsigned long bit_string = bit_string_lookup[i];
            curr = traverse_bit_string(root, bit_string);

            for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++) {
                if(curr->edge_block_entry[i].destination_vertex == 0) break;
                else if(curr->edge_block_entry[i].destination_vertex == destination_vertex) {
                        curr->edge_block_entry[i].destination_vertex = INFTY;
                        found = true;
                        break;
                }
            }
            if(found) break;
        }
    }
}

__global__ void compactionVertexCentric(unsigned long totalvertices, VertexDictionary *device_vertex_dictionary){
    unsigned long id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= totalvertices) return;

    EdgeBlock *curr = device_vertex_dictionary->edge_block_address[id];
    EdgeBlock *root = device_vertex_dictionary->edge_block_address[id];

    unsigned long src = device_vertex_dictionary->vertex_id[id];

    unsigned long bitString;
    unsigned long parent_bit_string;
    EdgeBlock *parent = NULL;
    unsigned int push_index_for_edge_queue;

    if(!curr) return;

    EdgeBlock *swapping_block = device_vertex_dictionary->last_insert_edge_block[id];

    unsigned long total_edge_blocks = device_vertex_dictionary->edge_block_count[id];

    long curr_edge_block_index = 0;
    long last_edge_block_index = total_edge_blocks - 1;
    long last_swap_offset = device_vertex_dictionary->last_insert_edge_offset[id];
    --last_swap_offset;

    while(curr_edge_block_index < last_edge_block_index){

        for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; ++i){
             unsigned long e = curr->edge_block_entry[i].destination_vertex;

             if(e != INFTY) continue;
             // deleted edge found
             int edge_swapped_flag = 0;

             while(edge_swapped_flag == 0){
                 if(curr_edge_block_index == last_edge_block_index) break;

                 if(last_swap_offset == -1){

                     push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1ULL);

                     push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                     d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                     swapping_block->lptr = NULL;
                     swapping_block->rptr = NULL;
                     device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                     swapping_block->active_edge_count = 0;
                     swapping_block = swapping_block->level_order_predecessor;

                     swapping_block->lptr = NULL;
                     swapping_block->rptr = NULL;
                     last_swap_offset = EDGE_BLOCK_SIZE - 1;

                     // freeing the parent to child relation
                     if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                     else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                     parent = traverse_bit_string(root, parent_bit_string);

                     if(last_edge_block_index & 1) parent->lptr = NULL;
                     else parent->rptr = NULL;

                     --last_edge_block_index;
                     device_vertex_dictionary->edge_block_count[id] -= 1;
                     if(curr_edge_block_index == last_edge_block_index) break;
                 } else{
                     while(last_swap_offset >= 0 && swapping_block->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {
                         swapping_block->active_edge_count -= 1;
                         device_vertex_dictionary->active_edge_count[id] -= 1;
                         --last_swap_offset;
                     }

                     if(last_swap_offset < 0){
                         push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1ULL);
                         push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                         d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                         swapping_block->lptr = NULL;
                         swapping_block->rptr = NULL;
                         device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                         swapping_block->active_edge_count = 0;
                         swapping_block = swapping_block->level_order_predecessor;
                         swapping_block->lptr = NULL;
                         swapping_block->rptr = NULL;
                         last_swap_offset = EDGE_BLOCK_SIZE - 1;

                         // freeing the parent to child relation
                         if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                         else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                         parent = traverse_bit_string(root, parent_bit_string);

                         if(last_edge_block_index & 1) parent->lptr = NULL;
                         else parent->rptr = NULL;

                         --last_edge_block_index;
                         device_vertex_dictionary->edge_block_count[id] -= 1;
                     }
                     else{
                         if(curr_edge_block_index == last_edge_block_index) break;

                         curr->edge_block_entry[i].destination_vertex = swapping_block->edge_block_entry[last_swap_offset].destination_vertex;
                         swapping_block->edge_block_entry[last_swap_offset].destination_vertex = 0;
                         --last_swap_offset;
                         swapping_block->active_edge_count -= 1;
                         device_vertex_dictionary->active_edge_count[id] -= 1;
                         edge_swapped_flag = 1;

                         if(last_swap_offset == -1){
                             push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1ULL);
                             push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
                             d_e_queue.edge_block_address[push_index_for_edge_queue] = swapping_block;
                             swapping_block->lptr = NULL;
                             swapping_block->rptr = NULL;
                             device_vertex_dictionary->active_edge_count[id] -= swapping_block->active_edge_count;
                             swapping_block->active_edge_count = 0;
                             swapping_block = swapping_block->level_order_predecessor;
                             swapping_block->lptr = NULL;
                             swapping_block->rptr = NULL;
                             last_swap_offset = EDGE_BLOCK_SIZE - 1;

                             // freeing the parent to child relation
                             if(last_edge_block_index & 1) parent_bit_string = bit_string_lookup[last_edge_block_index / 2];
                             else parent_bit_string = bit_string_lookup[last_edge_block_index / 2 - 1];

                             parent = traverse_bit_string(root, parent_bit_string);

                             if(last_edge_block_index & 1) parent->lptr = NULL;
                             else parent->rptr = NULL;

                             --last_edge_block_index;
                             device_vertex_dictionary->edge_block_count[id] -= 1;
                         }
                     }
                 }
             }

             if(curr_edge_block_index == last_edge_block_index) break;
        }

        if(curr_edge_block_index == last_edge_block_index) break;

        ++curr_edge_block_index;
        bitString = bit_string_lookup[curr_edge_block_index];
        curr = traverse_bit_string(root, bitString);
    }

    if(curr_edge_block_index == last_edge_block_index){
        last_swap_offset = EDGE_BLOCK_SIZE - 1;

        while(last_swap_offset >= 0 && (curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY || curr->edge_block_entry[last_swap_offset].destination_vertex == 0)) {

            if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {

                device_vertex_dictionary->active_edge_count[id] -= 1;
                curr->active_edge_count -= 1;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
            }
            --last_swap_offset;
        }

        long start = 0;

        while(start < EDGE_BLOCK_SIZE && start < last_swap_offset && last_swap_offset >= 0 && last_swap_offset < EDGE_BLOCK_SIZE){
            if(curr->edge_block_entry[start].destination_vertex != INFTY) ++start;

            else if(curr->edge_block_entry[last_swap_offset].destination_vertex == INFTY) {

                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
                device_vertex_dictionary->active_edge_count[id] -= 1;
                --last_swap_offset;
                --curr->active_edge_count;
            }
            else{

                curr->edge_block_entry[start].destination_vertex = curr->edge_block_entry[last_swap_offset].destination_vertex;
                curr->edge_block_entry[last_swap_offset].destination_vertex = 0;
                curr->active_edge_count -= 1;
                device_vertex_dictionary->active_edge_count[id] -= 1;
                --last_swap_offset;
            }
        }

        if(curr->edge_block_entry[start].destination_vertex == INFTY){
            curr->edge_block_entry[start].destination_vertex = 0;
            device_vertex_dictionary->active_edge_count[id] -= 1;
            --curr->active_edge_count;
        }
    }

    curr->lptr = NULL;
    curr->rptr = NULL;

    if(curr->active_edge_count == 0){
        push_index_for_edge_queue = atomicAdd(&(d_e_queue.rear), 1);
        push_index_for_edge_queue %= EDGE_PREALLOCATE_LIST_SIZE;
        d_e_queue.edge_block_address[push_index_for_edge_queue] = curr;
        device_vertex_dictionary->last_insert_edge_block[id] = curr->level_order_predecessor;
        device_vertex_dictionary->last_insert_edge_offset[id] = EDGE_BLOCK_SIZE - 1;
        device_vertex_dictionary->edge_block_count[id] -= 1;

        if(curr == root){

            curr->level_order_predecessor = NULL;
            device_vertex_dictionary->edge_block_address[id] = NULL;
            device_vertex_dictionary->active_edge_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_offset[id] = 0;
            device_vertex_dictionary->edge_block_count[id] = 0;
            device_vertex_dictionary->last_insert_edge_block[id] = NULL;
        }
    }
}

