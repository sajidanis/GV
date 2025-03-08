#include "pr_kernel.cuh"

__global__ void pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_prev_pr, float *d_curr_pr) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if ((id < vertex_size)) {
        unsigned long source_vertex = id;
        unsigned long edge_blocks = device_vertex_dictionary->edge_block_count[source_vertex];
        unsigned long edge_count = device_vertex_dictionary->active_edge_count[source_vertex];

        // printf("[%ld] => Edge blocks : %ld \t Edge count: %ld\n", id, edge_blocks, edge_count);

        EdgeBlock *parent = device_vertex_dictionary->edge_block_address[source_vertex];
        EdgeBlock *root;

        for (unsigned long i = 0; i < edge_blocks; i++) {
            unsigned long bit_string = bit_string_lookup[i];

            root = traverse_bit_string(parent, bit_string);
            // float page_factor = d_prev_pr[source_vertex] / edge_count;
            float page_factor = (edge_count > 0) ? d_prev_pr[source_vertex] / edge_count : 0.0f;

            for (unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++) {
                if (root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {
                    atomicAdd(&d_curr_pr[root->edge_block_entry[i].destination_vertex - 1], page_factor);
                }
            }
        }
    }
}

__global__ void pagerank_post_kernel(VertexDictionary *device_vertex_dictionary, size_t vertex_size, float damp, float normalized_damp, float *d_prev_pr, float *d_curr_pr, float *d_difference) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size){

        float new_pr = normalized_damp + (damp * d_curr_pr[id]);
        d_curr_pr[id] = new_pr;
        float diff = fabsf(d_curr_pr[id] - d_prev_pr[id]);
        d_difference[id] = diff;
        d_prev_pr[id] = d_curr_pr[id];
        d_curr_pr[id] = 0.0f;
    }

}

__global__ void dynamic_pageRank_kernel_HD(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size,unsigned long *d_affected_nodes, float *d_prev_pr, float *d_curr_pr) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if ((id < vertex_size) && d_affected_nodes[id] == 1) {
        unsigned long source_vertex = id;
        unsigned long edge_blocks = device_vertex_dictionary->edge_block_count[source_vertex];
        unsigned long edge_count = device_vertex_dictionary->active_edge_count[source_vertex];

        // printf("[%ld] => Edge blocks : %ld \t Edge count: %ld\n", id, edge_blocks, edge_count);

        EdgeBlock *root;

        for (unsigned long i = 0; i < edge_blocks; i++) {
            unsigned long bit_string = bit_string_lookup[i];
            EdgeBlock *parent = device_vertex_dictionary->edge_block_address[source_vertex];

            root = traverse_bit_string(parent, bit_string);
            float page_factor = (edge_count > 0) ? d_prev_pr[source_vertex] / edge_count : 0.0f;

            for (unsigned long i = 0; i < EDGE_BLOCK_SIZE; i++) {
                if (root->edge_block_entry[i].destination_vertex == 0)
                    break;
                else {
                    atomicAdd(&d_curr_pr[root->edge_block_entry[i].destination_vertex - 1], page_factor);
                }
            }
        }
    }
}

__global__ void pagerank_post_kernel_dynamic(VertexDictionary *device_vertex_dictionary, size_t vertex_size, float damp, float normalized_damp, float *d_prev_pr, float *d_curr_pr, float *d_difference, unsigned long *d_affected_nodes){

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size){

        float diff = fabsf(d_curr_pr[id] - d_prev_pr[id]);

        if(diff > 0.0001){
            d_affected_nodes[id] = 1;
        } else {
            d_affected_nodes[id] = 0;
            d_difference[id] = diff;
            return;
        }

        float new_pr = normalized_damp + (damp * d_curr_pr[id]);
        d_curr_pr[id] = new_pr;
        diff = fabsf(d_curr_pr[id] - d_prev_pr[id]);
        d_difference[id] = diff;
        d_prev_pr[id] = d_curr_pr[id];
        d_curr_pr[id] = 0.0f;
    }
}