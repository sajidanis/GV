#include "pr_kernel.cuh"

__global__ void pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_pageRankVector_1, float *d_pageRankVector_2, unsigned long *d_source_vector) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if ((id < vertex_size)) {
        unsigned long source_vertex = id;
        unsigned long edge_blocks = device_vertex_dictionary->edge_block_count[source_vertex];
        unsigned long edge_count = device_vertex_dictionary->active_edge_count[source_vertex];

        // printf("[%ld] => Edge blocks : %ld \t Edge count: %ld\n", id, edge_blocks, edge_count);

        for (unsigned long i = 0; i < edge_blocks; i++) {
            unsigned long bit_string = bit_string_lookup[i];
            EdgeBlock *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);
            float page_factor = d_pageRankVector_1[source_vertex] / edge_count;

            for (unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++) {
                if (root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {
                    atomicAdd(&d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1], page_factor);
                }
            }
        }
    }
}

__global__ void dynamic_pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size,unsigned long *d_affected_nodes, float *d_pageRankVector_1, float *d_pageRankVector_2) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if ((id < vertex_size) && d_affected_nodes[id] == 1) {
        unsigned long source_vertex = id;
        unsigned long edge_blocks = device_vertex_dictionary->edge_block_count[source_vertex];
        unsigned long edge_count = device_vertex_dictionary->active_edge_count[source_vertex];

        // printf("[%ld] => Edge blocks : %ld \t Edge count: %ld\n", id, edge_blocks, edge_count);

        for (unsigned long i = 0; i < edge_blocks; i++) {
            unsigned long bit_string = bit_string_lookup[i];
            EdgeBlock *root = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(root, bit_string);
            float page_factor = d_pageRankVector_1[source_vertex] / edge_count;

            for (unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++) {
                if (root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {
                    atomicAdd(&d_pageRankVector_2[root->edge_block_entry[i].destination_vertex - 1], page_factor);
                }
            }
        }
    }
}