#include "insert_kernels.cuh"

__global__ void batched_edge_inserts_EC(EdgeBlock *d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, VertexDictionary *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees)
{

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < batch_size)
    {
        unsigned long target_vertex = d_csr_edges[id];

        if (target_vertex != INFTY)
        {

            unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size);
            unsigned long active_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
            unsigned long new_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            EdgeBlock *root = NULL;
            EdgeBlock *base = NULL;

            if ((device_vertex_dictionary->edge_block_address[source_vertex] == NULL) || (batch_number == 0))
            {

                if (new_edge_block_count > 0)
                {

                    unsigned long index_counter = id - d_csr_offset[source_vertex];
                    unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
                    base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
                    root = base + current_edge_block_counter;
                    
                    if (!(index_counter % EDGE_BLOCK_SIZE))
                    {


                        unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
                        unsigned long length = 0;
                
                        unsigned long bit_string = bit_string_lookup[global_index_counter];

                        insert_edge_block_to_CBT_v2(NULL, bit_string, length, root, NULL, current_edge_block_counter, global_index_counter, active_edge_block_count, active_edge_block_count + new_edge_block_count, source_vertex, id);

                    }

                    unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
                    // return;

                    root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                    return;
                }
            }

            // below else is taken if it's the subsequent batch insert to an adjacency
            else if ((device_vertex_dictionary->edge_block_address[source_vertex] != NULL) && (batch_number))
            {
                unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
                unsigned long space_remaining = 0;
                if (last_insert_edge_offset != 0)
                    space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;
                unsigned long index_counter = id - d_csr_offset[source_vertex];

                // fill up newly allocated edge_blocks
                if (((index_counter >= space_remaining)) && (new_edge_block_count > 0))
                {

                    index_counter -= space_remaining;

                    // current_edge_block_counter value is 0 for the first new edge block
                    unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
                    base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
                    root = base + current_edge_block_counter;

                    if (!(index_counter % EDGE_BLOCK_SIZE))
                    {

                        unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
                        unsigned long length = 0;
                    
                        unsigned long bit_string = bit_string_lookup[global_index_counter];

                        insert_edge_block_to_CBT_v2(device_vertex_dictionary->edge_block_address[source_vertex], bit_string, length, root, device_vertex_dictionary->last_insert_edge_block[source_vertex], current_edge_block_counter, global_index_counter, active_edge_block_count, active_edge_block_count + new_edge_block_count, source_vertex, id);

                    }

                    unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
                    root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                }

                // fill up remaining space in last_insert_edge_block
                else
                {

                    if ((index_counter < space_remaining) && (space_remaining != EDGE_BLOCK_SIZE))
                    {

                        // traverse to last insert edge block
                        unsigned long edge_entry_index = index_counter + last_insert_edge_offset;
                        device_vertex_dictionary->last_insert_edge_block[source_vertex]->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;

                        return;
                    }
                }
            }
        }
    }
}


__global__ void batched_edge_inserts_EC_postprocessing(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees, unsigned long *d_prefix_sum_edge_blocks, unsigned long batch_number) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size){
        unsigned long target_vertex = d_csr_edges[id];
        if (1) {
            unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size + 1);
            unsigned long index_counter = id - d_csr_offset[source_vertex];

            if (!index_counter){

                unsigned long active_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
                unsigned long new_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

                EdgeBlock *base = pop_edge_block_address(d_prefix_sum_edge_blocks[vertex_size], d_prefix_sum_edge_blocks, source_vertex);

                if ((device_vertex_dictionary->edge_block_address[source_vertex] == NULL) || (batch_number == 0)) {
                    if (!batch_number)
                        device_vertex_dictionary->active_edge_count[source_vertex] = d_source_degrees[source_vertex];

                    device_vertex_dictionary->edge_block_address[source_vertex] = base;
                    device_vertex_dictionary->vertex_id[source_vertex] = source_vertex + 1;
                    device_vertex_dictionary->edge_block_count[source_vertex] = new_edge_block_count;
                
                    device_vertex_dictionary->last_insert_edge_block[source_vertex] = base + new_edge_block_count - 1;
                   
                    device_vertex_dictionary->last_insert_edge_offset[source_vertex] = d_source_degrees[source_vertex] % EDGE_BLOCK_SIZE;
                } else if ((device_vertex_dictionary->edge_block_address[source_vertex] != NULL) && (batch_number)) {

                    unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
                    unsigned long space_remaining = 0;
                    if (last_insert_edge_offset != 0)
                        space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;

                    device_vertex_dictionary->edge_block_count[source_vertex] += new_edge_block_count;
        
                    device_vertex_dictionary->active_edge_count[source_vertex] += d_source_degrees[source_vertex];

                    if (new_edge_block_count)
                        device_vertex_dictionary->last_insert_edge_block[source_vertex] = base + new_edge_block_count - 1;
                    device_vertex_dictionary->last_insert_edge_offset[source_vertex] = (d_source_degrees[source_vertex] - space_remaining) % EDGE_BLOCK_SIZE;
                }
            }
        }
    }
}

__global__ void update_edge_queue(unsigned long pop_count){
    d_e_queue.count -= (unsigned int)pop_count;

    if ((d_e_queue.front + (unsigned int)pop_count - 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.rear) {
        d_e_queue.front = -1;
        d_e_queue.rear = -1;
    }
    else
        d_e_queue.front = (d_e_queue.front + (unsigned int)pop_count) % EDGE_PREALLOCATE_LIST_SIZE;
}