#include "insert_kernels.cuh"

// __global__ void batched_edge_inserts_EC(EdgeBlock *d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, VertexDictionary *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_source_degrees) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if (id < batch_size) {
//         unsigned long target_vertex = d_csr_edges[id];

//         if (target_vertex != INFTY) {

//             unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size);
//             unsigned long active_edge_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
//             unsigned long new_edge_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

//             EdgeBlock *root = NULL;
//             EdgeBlock *base = NULL;

//             if ((device_vertex_dictionary->edge_block_address[source_vertex] == NULL) || (batch_number == 0)) {

//                 if (new_edge_block_count > 0) {
//                     unsigned long index_counter = id - d_csr_offset[source_vertex];
//                     unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
//                     base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
//                     root = base + current_edge_block_counter;
                    
//                     if (!(index_counter % EDGE_BLOCK_SIZE)) {
//                         unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
//                         unsigned long length = 0;
                
//                         unsigned long bit_string = bit_string_lookup[global_index_counter];
//                         root->src_vertex = source_vertex + 1;
                        
//                         insert_edge_block_to_CBT_v2(NULL, bit_string, length, root, NULL, current_edge_block_counter, global_index_counter, active_edge_block_count, active_edge_block_count + new_edge_block_count, source_vertex, id);

//                     }

//                     unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
//                     // return;

//                     root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
//                     atomicAdd(&(root->active_edge_count), 1);
//                     return;
//                 }
//             } else if ((device_vertex_dictionary->edge_block_address[source_vertex] != NULL) && (batch_number)) { 
//                 // below else is taken if it's the subsequent batch insert to an adjacency
                
//                 unsigned long last_insert_edge_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
//                 unsigned long space_remaining = 0;
//                 if (last_insert_edge_offset != 0)
//                     space_remaining = EDGE_BLOCK_SIZE - last_insert_edge_offset;
//                 unsigned long index_counter = id - d_csr_offset[source_vertex];

//                 // fill up newly allocated edge_blocks
//                 if (((index_counter >= space_remaining)) && (new_edge_block_count > 0)) {
                    
//                     index_counter -= space_remaining;

//                     // current_edge_block_counter value is 0 for the first new edge block
//                     unsigned long current_edge_block_counter = (index_counter / EDGE_BLOCK_SIZE);
//                     base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
//                     root = base + current_edge_block_counter;

//                     if (!(index_counter % EDGE_BLOCK_SIZE)) {

//                         unsigned long global_index_counter = active_edge_block_count + current_edge_block_counter;
//                         unsigned long length = 0;
                    
//                         unsigned long bit_string = bit_string_lookup[global_index_counter];
//                         root->src_vertex = source_vertex + 1;

//                         insert_edge_block_to_CBT_v2(device_vertex_dictionary->edge_block_address[source_vertex], bit_string, length, root, device_vertex_dictionary->last_insert_edge_block[source_vertex], current_edge_block_counter, global_index_counter, active_edge_block_count, active_edge_block_count + new_edge_block_count, source_vertex, id);

//                     }

//                     unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
//                     root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
//                     atomicAdd(&(root->active_edge_count), 1);
//                 } else { // fill up remaining space in last_insert_edge_block
//                     if ((index_counter < space_remaining) && (space_remaining != EDGE_BLOCK_SIZE)) {
//                         // traverse to last insert edge block
//                         unsigned long edge_entry_index = index_counter + last_insert_edge_offset;
//                         device_vertex_dictionary->last_insert_edge_block[source_vertex]->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;
                        
//                         atomicAdd(&(device_vertex_dictionary->last_insert_edge_block[source_vertex]->active_edge_count), 1);

//                         // printf("ID: %ld, edge_index: %ld, address of last edge block: %p\n", id, edge_entry_index, device_vertex_dictionary->last_insert_edge_block[source_vertex]);
//                         return;
//                     }
//                 }
//             }
//         }
//     }
// }


__global__ void batched_edge_inserts_EC_opt(
    EdgeBlock *d_edge_preallocate_list,
    unsigned long *d_edge_blocks_count_init,
    unsigned long total_edge_blocks_count_batch,
    unsigned long vertex_size,
    unsigned long edge_size,
    unsigned long *d_prefix_sum_edge_blocks,
    unsigned long thread_blocks,
    VertexDictionary *device_vertex_dictionary,
    unsigned long batch_number,
    unsigned long batch_size,
    unsigned long start_index_batch,
    unsigned long end_index_batch,
    unsigned long *d_csr_offset,
    unsigned long *d_csr_edges,
    unsigned long *d_source_degrees
) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = threadIdx.x / warpSize;
    unsigned int lane_id = threadIdx.x % warpSize;

    if (id >= batch_size) return;

    unsigned long target_vertex = d_csr_edges[id];
    if (target_vertex == INFTY) return;

    // Load source vertex via binary search (expensive, might consider storing precomputed)
    unsigned long source_vertex = device_binary_search(d_csr_offset, id, vertex_size);
    unsigned long src_offset = d_csr_offset[source_vertex];
    unsigned long index_counter = id - src_offset;
    unsigned long active_block_count = device_vertex_dictionary->edge_block_count[source_vertex];
    unsigned long new_block_count = d_prefix_sum_edge_blocks[source_vertex + 1] - d_prefix_sum_edge_blocks[source_vertex];

    bool is_first_batch = (batch_number == 0);
    bool has_existing_blocks = (device_vertex_dictionary->edge_block_address[source_vertex] != NULL);

    EdgeBlock *base = nullptr;
    EdgeBlock *root = nullptr;

    // Case 1: New allocation on first batch or vertex never had blocks
    if (!has_existing_blocks || is_first_batch) {
        if (new_block_count == 0) return;

        unsigned long block_index = index_counter / EDGE_BLOCK_SIZE;
        base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
        root = base + block_index;

        if ((index_counter % EDGE_BLOCK_SIZE) == 0) {
            unsigned long global_index = active_block_count + block_index;
            unsigned long length = 0;
            unsigned long bit_string = bit_string_lookup[global_index];

            // One thread per warp inserts block, warp sync reduces duplication
            if (lane_id == 0) {
                root->src_vertex = source_vertex + 1;
                insert_edge_block_to_CBT_v2(
                    NULL, bit_string, length, root, NULL,
                    block_index, global_index,
                    active_block_count, active_block_count + new_block_count,
                    source_vertex, id
                );
            }
        }

        unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
        root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;

        // Warp aggregation for atomicAdd
        unsigned int vote = __ballot_sync(0xffffffff, true);
        if (lane_id == 0) {
            unsigned int n = __popc(vote);
            atomicAdd(&(root->active_edge_count), n);
        }
    }
    // Case 2: Appending to existing blocks in further batches
    else {
        unsigned long last_offset = device_vertex_dictionary->last_insert_edge_offset[source_vertex];
        unsigned long space_remaining = (last_offset > 0) ? (EDGE_BLOCK_SIZE - last_offset) : 0;

        if ((index_counter < space_remaining) && (space_remaining != EDGE_BLOCK_SIZE)) {
            // Fill last edge block
            unsigned long edge_index = index_counter + last_offset;
            EdgeBlock *last_block = device_vertex_dictionary->last_insert_edge_block[source_vertex];
            last_block->edge_block_entry[edge_index].destination_vertex = target_vertex;

            // Warp aggregation for atomicAdd
            unsigned int vote = __ballot_sync(0xffffffff, true);
            if (lane_id == 0) {
                unsigned int n = __popc(vote);
                atomicAdd(&(last_block->active_edge_count), n);
            }
        } else if (new_block_count > 0) {
            // Fill new edge blocks
            index_counter -= space_remaining;
            unsigned long block_index = index_counter / EDGE_BLOCK_SIZE;
            base = pop_edge_block_address(total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, source_vertex);
            root = base + block_index;

            if ((index_counter % EDGE_BLOCK_SIZE) == 0) {
                unsigned long global_index = active_block_count + block_index;
                unsigned long length = 0;
                unsigned long bit_string = bit_string_lookup[global_index];

                if (lane_id == 0) {
                    root->src_vertex = source_vertex + 1;
                    insert_edge_block_to_CBT_v2(
                        device_vertex_dictionary->edge_block_address[source_vertex], bit_string, length, root,
                        device_vertex_dictionary->last_insert_edge_block[source_vertex],
                        block_index, global_index,
                        active_block_count, active_block_count + new_block_count,
                        source_vertex, id
                    );
                }
            }

            unsigned long edge_entry_index = index_counter % EDGE_BLOCK_SIZE;
            root->edge_block_entry[edge_entry_index].destination_vertex = target_vertex;

            // Warp-level aggregated atomic add
            unsigned int vote = __ballot_sync(0xffffffff, true);
            if (lane_id == 0) {
                unsigned int n = __popc(vote);
                atomicAdd(&(root->active_edge_count), n);
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
    
    printf("Queue front is %lu(%p) and rear is %lu(%p)\n", d_e_queue.front, d_e_queue.edge_block_address[d_e_queue.front], d_e_queue.rear, d_e_queue.edge_block_address[d_e_queue.rear]);
    printf("Queue count is %u\n", d_e_queue.count);
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
                        d_csr_edges[d_csr_offset[source_vertex] + (index_counter % input_batch_degree)] = INFTY;
                        // root->edge_block_entry[i].destination_vertex = INFTY;
                        // device_vertex_dictionary->active_edge_count[source_vertex]--;
                        // atomicDec(&(device_vertex_dictionary->active_edge_count[source_vertex]), INFTY);
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

__global__ void device_sorting_post(){
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < d_e_queue.front) {
        EdgeBlock *root = d_e_queue.edge_block_address[id];
        unsigned long active_edge_count = root->active_edge_count;
        unsigned long edge_block_size = EDGE_BLOCK_SIZE;

        // thrust::sort(thrust::device, root->edge_block_entry, root->edge_block_entry + active_edge_count);
        thrust::sort(thrust::device, root->edge_block_entry, root->edge_block_entry + active_edge_count);
        
    }
}

__global__ void cub_sort_edge_blocks() {
    unsigned long long idx = blockIdx.x;
    if (idx >= d_e_queue.front) return;

    EdgeBlock* block = d_e_queue.edge_block_address[idx];
    const int tid = threadIdx.x;

    if (tid >= block->active_edge_count) return;

    using KeyT = unsigned long long;
    using ValueT = Edge;

    constexpr int BLOCK_THREADS = EDGE_BLOCK_SIZE;
    constexpr int ITEMS_PER_THREAD = 1;

    typedef cub::BlockRadixSort<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, ValueT> BlockRadixSortT;
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

    KeyT keys[ITEMS_PER_THREAD];
    ValueT values[ITEMS_PER_THREAD];

    keys[0] = block->edge_block_entry[tid].destination_vertex;
    values[0] = block->edge_block_entry[tid];

    // __syncthreads(); // barrier before sort
    BlockRadixSortT(temp_storage).Sort(keys, values);
    // __syncthreads(); // barrier after sort

    block->edge_block_entry[tid] = values[0];
}

__global__ void warp_bitonic_edge_sort() {
    unsigned long long block_idx = blockIdx.x;
    if (block_idx >= d_e_queue.front) return;

    EdgeBlock *block = d_e_queue.edge_block_address[block_idx];
    int tid = threadIdx.x;

    __shared__ Edge shared_edges[EDGE_BLOCK_SIZE];
    __shared__ int valid_count;

    Edge edge = block->edge_block_entry[tid];
    bool is_valid = (edge.destination_vertex != 0 && edge.destination_vertex != INFTY);

    // Phase 1: Warp-level Compaction
    unsigned int mask = __ballot_sync(0xFFFFFFFF, is_valid);
    int pos = __popc(mask & ((1U << (tid % 32)) - 1));
    int warp_id = tid / 32;

    __shared__ int warp_offsets[8];
    if ((tid % 32) == 0) {
        int total = __popc(mask);
        warp_offsets[warp_id] = atomicAdd(&valid_count, total);
    }
    __syncthreads();

    if (is_valid) {
        int global_pos = warp_offsets[warp_id] + pos;
        shared_edges[global_pos] = edge;
    }

    __syncthreads();

    // Phase 2: Bitonic sort only on valid_count elements
    int vcount = valid_count;
    for (int k = 2; k <= EDGE_BLOCK_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid && tid < vcount && ixj < vcount) {
                Edge a = shared_edges[tid];
                Edge b = shared_edges[ixj];
                if ((tid & k) == 0) {
                    if (a.destination_vertex > b.destination_vertex) {
                        shared_edges[tid] = b;
                        shared_edges[ixj] = a;
                    }
                } else {
                    if (a.destination_vertex < b.destination_vertex) {
                        shared_edges[tid] = b;
                        shared_edges[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Phase 3: Pad rest with INFTY
    if (tid >= valid_count && tid < EDGE_BLOCK_SIZE) {
        shared_edges[tid].destination_vertex = INFTY;
    }

    __syncthreads();

    // Phase 4: Write back
    block->edge_block_entry[tid] = shared_edges[tid];
    if (tid == 0) {
        block->active_edge_count = valid_count;
    }
}
