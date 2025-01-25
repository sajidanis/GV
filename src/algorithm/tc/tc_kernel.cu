#include "tc_kernel.cuh"

__global__ void triangle_counting_kernel_VC(VertexDictionary *device_vertex_dictionary, unsigned long vertex_size, float *d_triangleCount, unsigned long *d_csr_offset, unsigned long *d_csr_edges){
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size) {
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