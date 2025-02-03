#include "util_kernels.cuh"

__global__ void printDeviceVertexDictionary_kernel(size_t vertexSize, VertexDictionary* vertexDictionary) {
    for (int id = 0; id < vertexSize; id++) {
        printf("\tVertex [%d]: ", vertexDictionary->vertex_id[id]);
        size_t edge_blocks_count = vertexDictionary->edge_block_count[id];
        size_t activeEdges = vertexDictionary->active_edge_count[id];

        EdgeBlock *root = vertexDictionary->edge_block_address[id];

        for(unsigned long i = 0; i < edge_blocks_count; i++) {
            unsigned long bit_string = bit_string_lookup[i];
    
            root = traverse_bit_string(root, bit_string);

            for(unsigned long j = 0; j < EDGE_BLOCK_SIZE; j++) {
                if(root->edge_block_entry[j].destination_vertex == 0) break;
                if(root->edge_block_entry[j].destination_vertex == INFTY) continue;

                printf(" %d ", root->edge_block_entry[j].destination_vertex);
            }
        }
        printf("\n");
    }
}