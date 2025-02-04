#include "insert_kernels.cuh"

__global__ void batched_edge_inserts_EC(EdgeBlock *d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, VertexDictionary *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < batch_size) {
        unsigned long target_vertex = d_csr_edges[id];

        if (target_vertex != INFTY) {

            unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size);
            unsigned long active_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
            unsigned long new_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

            EdgeBlock *root = NULL;
            EdgeBlock *base = NULL;

            if ((device_vertex_dictionary->edge_block_address[source_vertex] == NULL) || (batch_number == 0)) {

                if (new_edge_block_count > 0) {
                    unsigned long index_counter = id - d_csr_offset[source_vertex];
                    unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
                    base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
                    root = base + current_edge_block_counter;
                    
                    if (!(index_counter % EDGE_BLOCK_SIZE)) {
                        unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
                        unsigned long length = 0;
                
                        unsigned long bit_string = bit_string_lookup[global_index_counter];

                        insert_edge_block_to_CBT_v2(NULL, bit_string, length, root, NULL, current_edge_block_counter, global_index_counter, active_edge_block_count, active_edge_block_count + new_edge_block_count, source_vertex, id);

                    }

                    unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
                    // return;

                    root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                    atomicAdd(&(root->active_edge_count), 1);
                    return;
                }
            } else if ((device_vertex_dictionary->edge_block_address[source_vertex] != NULL) && (batch_number)) { 
                // below else is taken if it's the subsequent batch insert to an adjacency
                
                unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
                unsigned long space_remaining = 0;
                if (last_insert_edge_offset != 0)
                    space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;
                unsigned long index_counter = id - d_csr_offset[source_vertex];

                // fill up newly allocated edge_blocks
                if (((index_counter >= space_remaining)) && (new_edge_block_count > 0)) {
                    
                    index_counter -= space_remaining;

                    // current_edge_block_counter value is 0 for the first new edge block
                    unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
                    base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
                    root = base + current_edge_block_counter;

                    if (!(index_counter % EDGE_BLOCK_SIZE)) {

                        unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
                        unsigned long length = 0;
                    
                        unsigned long bit_string = bit_string_lookup[global_index_counter];

                        insert_edge_block_to_CBT_v2(device_vertex_dictionary->edge_block_address[source_vertex], bit_string, length, root, device_vertex_dictionary->last_insert_edge_block[source_vertex], current_edge_block_counter, global_index_counter, active_edge_block_count, active_edge_block_count + new_edge_block_count, source_vertex, id);

                    }

                    unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
                    root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                    atomicAdd(&(root->active_edge_count), 1);
                } else { // fill up remaining space in last_insert_edge_block
                    if ((index_counter < space_remaining) && (space_remaining != EDGE_BLOCK_SIZE)) {
                        // traverse to last insert edge block
                        unsigned long edge_entry_index = index_counter + last_insert_edge_offset;
                        device_vertex_dictionary->last_insert_edge_block[source_vertex]->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                        
                        atomicAdd(&(device_vertex_dictionary->last_insert_edge_block[source_vertex]->active_edge_count), 1);

                        // printf("ID: %ld, edge_index: %ld, address of last edge block: %p\n", id, edge_entry_index, device_vertex_dictionary->last_insert_edge_block[source_vertex]);
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

__global__ void device_remove_batch_duplicates(unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < batch_size) {

        unsigned long source = device_binary_search((unsigned long *)d_csr_offset, id, vertex_size + 1);
        unsigned long index_counter = id - d_csr_offset[source];

        unsigned long start_index = id;
        unsigned long end_index = d_csr_offset[source + 1];
        unsigned long index = start_index;
        unsigned long prev_value = d_csr_edges[index++];

        // removing self-loops and duplicate edges
        for (unsigned long i = start_index + 1; i < end_index; i++){

            if ((d_csr_edges[i] == d_csr_edges[id]) || (d_csr_edges[i] == (source + 1))) {
                // if((d_csr_edges[i] == d_csr_edges[id])) {
                d_csr_edges[i] = INFTY;
            }
        }
    }
}

__global__ void find_affected_nodes(unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_affected_nodes){
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size){
        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        if (start_index < end_index){
            d_affected_nodes[id] = 1;
            for (unsigned long i = start_index; i < end_index; i++){
                if (d_csr_edges[i] != INFTY){
                    d_affected_nodes[d_csr_edges[i]] = 1;
                }
            }
        }
    }
}

__global__ void device_update_source_degrees(unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size){

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        if (start_index < end_index){

            for (unsigned long i = start_index; i < end_index; i++){

                if (d_csr_edges[i] == INFTY)
                    d_source_degrees[id]--;
            }

            unsigned long l_index = start_index;
            unsigned long r_index = end_index - 1;

            while (l_index < r_index){

                while ((d_csr_edges[l_index] != INFTY) && (l_index < r_index))
                    l_index++;
                while ((d_csr_edges[r_index] == INFTY) && (l_index < r_index))
                    r_index--;

                // printf("ID=%lu, l_index=%lu, r_index=%lu\n", id, l_index, r_index);

                if (l_index < r_index){
                    // printf("ID=%lu, l_index=%lu, r_index=%lu\n", id, l_index, r_index);
                    unsigned long temp = d_csr_edges[l_index];
                    d_csr_edges[l_index] = d_csr_edges[r_index];
                    d_csr_edges[r_index] = temp;
                    // d_source_degrees[id]--;
                }
            }
        }
    }
}

__global__ void batched_delete_preprocessing_EC_LD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees, unsigned long *d_source_vector) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size){
        unsigned long vertex_edge_blocks = device_vertex_dictionary->edge_block_count[id];
        unsigned long batch_degree = d_csr_offset[id + 1] - d_csr_offset[id];

        d_source_vector[id] = vertex_edge_blocks * batch_degree;
    }
    
}

__global__ void batched_delete_kernel_EC_LD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if ((id < ceil((double)d_source_vector[vertex_size]))) {

        unsigned long source_vertex = device_binary_search(d_source_vector, id, vertex_size + 1);
        unsigned long input_batch_degree = d_csr_offset[source_vertex + 1] - d_csr_offset[source_vertex];
        unsigned long index_counter = id - d_source_vector[source_vertex];
        // unsigned long thread_count_source = d_source_vector[source_vertex + 1] - d_source_vector[source_vertex];
        unsigned long edge_block_index = index_counter / input_batch_degree;
        unsigned long target_vertex = d_csr_edges[d_csr_offset[source_vertex] + (index_counter % input_batch_degree)];

        if (target_vertex != INFTY) {
            unsigned long bit_string = bit_string_lookup[edge_block_index];
            EdgeBlock *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);

            for (unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++){

                if (root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else{
                    if ((root->edge_block_entry[i].destination_vertex == target_vertex)) {
                        root->edge_block_entry[i].destination_vertex = INFTY;
                        // device_vertex_dictionary->active_edge_count[source_vertex]--;
                        atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                        // root->active_edge_count--;
                        break;
                    }
                }
            }
        }
    }
}

__global__ void batched_delete_kernel_EC_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_vector){
    
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if ((id < ceil((double)batch_size))){
        unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size + 1);

        unsigned long target_vertex = d_csr_edges[id];

        if (target_vertex != INFTY){
            unsigned long source_degree_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];

            for (unsigned long z = 0; z < source_degree_edge_block_count; z++){
                unsigned long bit_string = bit_string_lookup[z];
                EdgeBlock *root = device_vertex_dictionary->edge_block_address[source_vertex];

                root = traverse_bit_string(root, bit_string);

                if ((root == NULL)){
                    printf("null hit at id=%lu, source_vertex=%lu, target_vertex=%lu, edge_block_count=%lu, counter=%lu, at GV=%lu\n", id, source_vertex, target_vertex, source_degree_edge_block_count, z, device_vertex_dictionary->edge_block_count[source_vertex]);
                   
                }

                for (unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++){

                    if (root->edge_block_entry[i].destination_vertex == 0)
                        break;
                    else {
                        if ((root->edge_block_entry[i].destination_vertex == target_vertex)) {
                            root->edge_block_entry[i].destination_vertex = INFTY;
                            // device_vertex_dictionary->active_edge_count[source_vertex]--;
                            atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
                            // root->active_edge_count--;
                            break;
                        }
                    }
                }
            }
        }
    }
}

__global__ void device_insert_preprocessing(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long batch_size, unsigned long *d_source_degrees, unsigned long batch_number, unsigned long *d_edge_blocks_count)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size) {
        if (d_source_degrees[id]) {
            unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[id];
            unsigned long space_remaining = 0;
            if (last_insert_edge_offset)
                space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;
            
            // printf("Source Degree: %ld, Space Remaining: %ld, Last Insert Offset: %ld\n", d_source_degrees[id], space_remaining, last_insert_edge_offset);

            unsigned long edge_blocks;
            if (batch_number != 0) {
                if (space_remaining == 0) {
                    edge_blocks = ceil(double(d_source_degrees[id]) / EDGE_BLOCK_SIZE);
                }
                else if (d_source_degrees[id] >= space_remaining) {
                    edge_blocks = ceil(double(d_source_degrees[id] - space_remaining) / EDGE_BLOCK_SIZE);
                }
                else {
                    edge_blocks = 0;
                }
            }
            else {
                edge_blocks = ceil(double(d_source_degrees[id]) / EDGE_BLOCK_SIZE);
            }
            d_edge_blocks_count[id] = edge_blocks;
            // printf("Number of edge blocks required by Id: %ld == %ld\n", id, edge_blocks);
        } else
            d_edge_blocks_count[id] = 0;
    }
}
