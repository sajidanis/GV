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

    for(size_t idx = csr_start; idx < csr_end; idx++) {
        unsigned long destination_vertex = d_csr_edges[idx];
        
        if(destination_vertex == INFTY) continue;

        found = false;

        for(unsigned long i = 0; i < edge_blocks_count; i++) {
            unsigned long bit_string = bit_string_lookup[i];
            EdgeBlock *root = device_vertex_dictionary->edge_block_address[id];

            root = traverse_bit_string(root, bit_string);

            for(unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++) {
                if(root->edge_block_entry[i].destination_vertex == 0) break;
                else if(root->edge_block_entry[i].destination_vertex == destination_vertex) {
                        root->edge_block_entry[i].destination_vertex = INFTY;
                        edge_count--;
                        found = true;
                        break;
                }
            }
            if(found) break;
        }
    }

    device_vertex_dictionary->active_edge_count[id] = edge_count;
}