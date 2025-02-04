#include "util_kernels.cuh"

__global__ void build_bit_string_lookup() {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < BIT_STRING_LOOKUP_SIZE){

        unsigned long length = 0;
        // value passed to traversal string is (id + 1) since we count edge blocks from 1
        bit_string_lookup[id] = traversal_string(id + 1, &length); //ANIS-Cmments { What is the use of the length parameter}
    }
}

__global__ void printDeviceVertexDictionary_kernel(size_t vertexSize, VertexDictionary* vertexDictionary) {

    vertexSize = vertexSize >= 10 ? 10 : vertexSize;

    for (int id = 0; id < vertexSize; id++) {
        
        size_t edge_blocks_count = vertexDictionary->edge_block_count[id];
        size_t activeEdges = vertexDictionary->active_edge_count[id];

        EdgeBlock *root = vertexDictionary->edge_block_address[id];
        EdgeBlock *curr;

        if(root!=NULL) printf("\tVertex [%ld]_[%ld] -> ", vertexDictionary->vertex_id[id], activeEdges);

        printf(" [%p]:", root);

        for(unsigned long i = 0; i < edge_blocks_count; i++) {
            unsigned long bit_string = bit_string_lookup[i];

            curr = traverse_bit_string(root, bit_string);

            printf(" [%ld-%ld]{ ", bit_string, curr->active_edge_count);

            for(unsigned long j = 0; j < EDGE_BLOCK_SIZE; j++) {
                if(curr->edge_block_entry[j].destination_vertex == 0) break;
                // if(curr->edge_block_entry[j].destination_vertex == INFTY) continue;

                printf(" %d ", curr->edge_block_entry[j].destination_vertex);
            }

            printf(" }  ");
        }
        printf("\n");
    }
}