#include "tc_kernel.cuh"

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

    // if(u < 10){
    //     printf("Vertex %ld has %ld edge blocks\n", u, u_edge_blocks);
    //     printf("Address of edge blocks: %p\n", device_vertex_dictionary->edge_block_address[u]);
    // }

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
                                // printf("U: %ld, V: %ld, W: %ld, Cand: %ld\n", src, v, w, candidate);
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

__global__ void tc_kernel_VC_sorted(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangle_count){
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

    for (unsigned long i_block = 0; i_block < u_edge_blocks; i_block++) {
        u_block = traverse_bit_string(u_block, bit_string_lookup[i_block]);
        for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
            unsigned long v = u_block->edge_block_entry[i].destination_vertex;
            if (v <= u || v == INFTY || v == 0) continue;

            // Get v's edge blocks
            EdgeBlock *v_block = device_vertex_dictionary->edge_block_address[v - 1];
            unsigned long v_edge_blocks = device_vertex_dictionary->edge_block_count[v - 1];

            for (unsigned long j_block = 0; j_block < v_edge_blocks; j_block++) {
                v_block = traverse_bit_string(v_block, bit_string_lookup[j_block]);
                for (int j = 0; j < EDGE_BLOCK_SIZE; j++) {
                    unsigned long w = v_block->edge_block_entry[j].destination_vertex;
                    if (w <= v || w == INFTY || w == 0) continue;

                    // Set intersection: check if w has u as neighbor (since edge list is sorted)
                    EdgeBlock *w_block = device_vertex_dictionary->edge_block_address[w - 1];
                    unsigned long w_edge_blocks = device_vertex_dictionary->edge_block_count[w - 1];

                    // Merge-style set intersection (u is being searched in w's neighbor list)
                    unsigned long k_block = 0;
                    bool found = false;

                    while (k_block < w_edge_blocks && !found) {
                        w_block = traverse_bit_string(w_block, bit_string_lookup[k_block]);
                        for (int k = 0; k < EDGE_BLOCK_SIZE; k++) {
                            unsigned long candidate = w_block->edge_block_entry[k].destination_vertex;

                            if (candidate == INFTY || candidate == 0) break;  // End of valid entries
                            if (candidate > src) break;                       // Since sorted, no point in continuing
                            if (candidate == src) {
                                found = true;
                                break;
                            }
                        }
                        k_block++;
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

__global__ void dynamic_tc_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangle_count, unsigned long *d_affected_nodes){
    unsigned long u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= vertex_size || d_affected_nodes[u] == 0) return;
 
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

    // if(u < 10){
    //     printf("Vertex %ld has %ld edge blocks\n", u, u_edge_blocks);
    //     printf("Address of edge blocks: %p\n", device_vertex_dictionary->edge_block_address[u]);
    // }

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
                                // printf("U: %ld, V: %ld, W: %ld, Cand: %ld\n", src, v, w, candidate);
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