#include "tc_kernel.cuh"

__global__ void triangle_counting_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_csr_offset, unsigned long *d_csr_edges){
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size) {
        d_triangleCount[id] = 0;
        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        unsigned long start_index_second_vertex = (tc_device_binary_search(d_csr_edges + start_index, id, end_index - start_index));

        for (unsigned long i = start_index_second_vertex + d_csr_offset[id]; i < end_index; i++) {

            unsigned long second_vertex = d_csr_edges[i] - 1;

            unsigned long first_adjacency_size = end_index - start_index;
            unsigned long second_adjacency_size = d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex];

            if (first_adjacency_size <= second_adjacency_size) {

                unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + start_index, second_vertex, end_index - start_index);

                for (unsigned long j = start_index_third_vertex + d_csr_offset[id]; j < end_index; j++) {

                    unsigned long third_vertex = d_csr_edges[j] - 1;

                    unsigned long target_vertex_index;

                    // if(second_vertex != third_vertex) {
                    target_vertex_index = tc_final_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], third_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);
                    if (target_vertex_index) {
                        // atomicAdd(&total_triangles, 1);
                        atomicAdd(&d_triangleCount[id], 1);
                        atomicAdd(&d_triangleCount[second_vertex], 1);
                        atomicAdd(&d_triangleCount[third_vertex], 1);
                        // break;
                    }
                }
            }

            else {

                unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], second_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);

                for (unsigned long j = start_index_third_vertex + d_csr_offset[second_vertex]; j < d_csr_offset[second_vertex + 1]; j++) {

                    unsigned long third_vertex = d_csr_edges[j] - 1;
                    unsigned long target_vertex_index;

                    // if(second_vertex != third_vertex) {
                    target_vertex_index = tc_final_device_binary_search(d_csr_edges + start_index, third_vertex, end_index - start_index);

                    if (target_vertex_index) {
                        // atomicAdd(&total_triangles, 1);
                        atomicAdd(&d_triangleCount[id], 1);
                        atomicAdd(&d_triangleCount[second_vertex], 1);
                        atomicAdd(&d_triangleCount[third_vertex], 1);
                    }
                }
            }
        }
    }
}

__global__ void dynamic_triangle_counting_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_affected_nodes){
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size && d_affected_nodes[id] == 1) {

        d_triangleCount[id] = 0;

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];

        unsigned long start_index_second_vertex = (tc_device_binary_search(d_csr_edges + start_index, id, end_index - start_index));

        for (unsigned long i = start_index_second_vertex + d_csr_offset[id]; i < end_index; i++) {

            unsigned long second_vertex = d_csr_edges[i] - 1;

            unsigned long first_adjacency_size = end_index - start_index;
            unsigned long second_adjacency_size = d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex];

            if (first_adjacency_size <= second_adjacency_size) {

                unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + start_index, second_vertex, end_index - start_index);

                for (unsigned long j = start_index_third_vertex + d_csr_offset[id]; j < end_index; j++) {

                    unsigned long third_vertex = d_csr_edges[j] - 1;

                    unsigned long target_vertex_index;

                    // if(second_vertex != third_vertex) {
                    target_vertex_index = tc_final_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], third_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);
                    if (target_vertex_index) {
                        // atomicAdd(&total_triangles, 1);
                        atomicAdd(&d_triangleCount[id], 1);
                        atomicAdd(&d_triangleCount[second_vertex], 1);
                        atomicAdd(&d_triangleCount[third_vertex], 1);
                        // break;
                    }
                }
            }

            else {

                unsigned long start_index_third_vertex = tc_device_binary_search(d_csr_edges + (unsigned long)d_csr_offset[second_vertex], second_vertex, d_csr_offset[second_vertex + 1] - d_csr_offset[second_vertex]);

                for (unsigned long j = start_index_third_vertex + d_csr_offset[second_vertex]; j < d_csr_offset[second_vertex + 1]; j++) {

                    unsigned long third_vertex = d_csr_edges[j] - 1;
                    unsigned long target_vertex_index;

                    // if(second_vertex != third_vertex) {
                    target_vertex_index = tc_final_device_binary_search(d_csr_edges + start_index, third_vertex, end_index - start_index);

                    if (target_vertex_index) {
                        // atomicAdd(&total_triangles, 1);
                        atomicAdd(&d_triangleCount[id], 1);
                        atomicAdd(&d_triangleCount[second_vertex], 1);
                        atomicAdd(&d_triangleCount[third_vertex], 1);
                    }
                }
            }
        }
    }
}

__global__ void tc_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangle_count){
    unsigned long u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= vertex_size) return;

    unsigned long edge_count = device_vertex_dictionary->active_edge_count[u];
    unsigned long src = device_vertex_dictionary->vertex_id[u];

    if(edge_count == 0) {
        d_triangle_count[u] = 0.0f;
        return;
    }

    float count = 0.0f;

    // Get u's edge blocks
    EdgeBlock *u_block = device_vertex_dictionary->edge_block_address[u];
    unsigned long u_edge_blocks = device_vertex_dictionary->edge_block_count[u];

    printf("Vertex %ld has %ld edge blocks\n", u, u_edge_blocks);

    // First pass: collect all neighbors of u
    for (unsigned long i_block = 0; i_block < u_edge_blocks; i_block++) {
        u_block = traverse_bit_string(u_block, bit_string_lookup[i_block]);
        for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
            unsigned long v = u_block->edge_block_entry[i].destination_vertex;
            // Only consider v > u to avoid duplicate counting
            if (v <= u || v == INFTY || v == 0) continue;

            

            // Get v's edge blocks
            EdgeBlock *v_block = device_vertex_dictionary->edge_block_address[v-1];
            unsigned long v_edge_blocks = device_vertex_dictionary->edge_block_count[v-1];

            // Check v's neighbors
            for (unsigned long j_block = 0; j_block < v_edge_blocks; j_block++) {
                v_block = traverse_bit_string(v_block, bit_string_lookup[j_block]);
                for (int j = 0; j < EDGE_BLOCK_SIZE; j++) {
                    unsigned long w = v_block->edge_block_entry[j].destination_vertex;
                    
                    // Only consider w > v to avoid duplicates
                    if (w <= v || w == INFTY || w == 0) continue;
                    
                    // Check if u exists in w's neighbors
                    EdgeBlock *check_block = device_vertex_dictionary->edge_block_address[w-1];
                    unsigned long check_blocks = device_vertex_dictionary->edge_block_count[w-1];
                    bool found = false;

                    // Linear search through w's neighbors
                    for (unsigned long k_block = 0; k_block < check_blocks && !found; k_block++) {
                        check_block = traverse_bit_string(check_block, bit_string_lookup[k_block]);
                        for (int k = 0; k < EDGE_BLOCK_SIZE; k++) {
                            unsigned long candidate = check_block->edge_block_entry[k].destination_vertex;
                            
                            if (candidate == src) {
                                printf("U: %ld, V: %ld, W: %ld, Cand: %ld\n", src, v, w, candidate);
                                found = true;
                                break;
                            }
                        }
                    }
                    if (found) {
                        count += 1.0f;
                    }
                }
            }
        }
    }

    d_triangle_count[u] = count;
}