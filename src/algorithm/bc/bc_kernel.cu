#include "bc_kernel.cuh"

__global__ void initializeArrays(int n, int *sigma, float *delta, int *dist) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        sigma[tid] = 0;      // Number of shortest paths
        delta[tid] = 0.0f;   // Dependency values
        dist[tid] = -1;     // Distance array initialized to infinity
    }
}

__global__ void initbfsKernel(int source, int *sigma, int *dist, int *frontier, int *frontier_size) {
    dist[source] = 0;
    sigma[source] = 1;
    frontier[0] = source;
    *frontier_size = 1;
}   

__global__ void bfsKernel(VertexDictionary *d_vertex_dict, int *sigma, int *dist, int *frontier, int *frontier_size, int *next_frontier, int *next_frontier_size) {

    __shared__ unsigned long shared_queue[THREADS_PER_BLOCK];

    unsigned long tid = threadIdx.x;

    int sharedSize = min(THREADS_PER_BLOCK, *frontier_size - blockIdx.x * THREADS_PER_BLOCK);

    if (tid < sharedSize) {

        shared_queue[tid] = frontier[blockIdx.x * THREADS_PER_BLOCK + tid];
    }
    __syncthreads();
    
    for (int i = tid; i < sharedSize; i += blockDim.x) {
        unsigned long curr_vertex = shared_queue[i];

        unsigned long edge_blocks = d_vertex_dict->edge_block_count[curr_vertex];

        // Going to each neighbour of curr_vertex
        EdgeBlock *root = d_vertex_dict->edge_block_address[curr_vertex];
        EdgeBlock *curr;

        for(int j = 0 ; j < edge_blocks ; j++){
            unsigned long bit_string = bit_string_lookup[j];
            curr = traverse_bit_string(root, bit_string);
            for (unsigned long k = 0; k < EDGE_BLOCK_SIZE; k++){
                int v = curr->edge_block_entry[k].destination_vertex;
                if (atomicCAS(&dist[v], -1, dist[curr_vertex] + 1) == -1) {
                    next_frontier[atomicAdd(next_frontier_size, 1)] = v;
                }
                if (dist[v] == dist[curr_vertex] + 1) {
                    atomicAdd(&sigma[v], sigma[curr_vertex]);
                }
            }
        }
    }
}

__global__ void bfsKernelSimplified(VertexDictionary *d_vertex_dict, int *sigma, int *dist, int *frontier, int *frontier_size, int *next_frontier, int *next_frontier_size){
    unsigned long tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < *frontier_size){
        unsigned long curr_vertex = frontier[tid];

        unsigned long edge_blocks = d_vertex_dict->edge_block_count[curr_vertex];

        // Going to each neighbour of curr_vertex
        EdgeBlock *root = d_vertex_dict->edge_block_address[curr_vertex];
        EdgeBlock *curr;

        for(int j = 0 ; j < edge_blocks ; j++){
            unsigned long bit_string = bit_string_lookup[j];
            curr = traverse_bit_string(root, bit_string);

            for (unsigned long k = 0; k < EDGE_BLOCK_SIZE; k++){
                int v = curr->edge_block_entry[k].destination_vertex;
                if (atomicCAS(&dist[v], -1, dist[curr_vertex] + 1) == -1) {
                    next_frontier[atomicAdd(next_frontier_size, 1)] = v;
                }
                if (dist[v] == dist[curr_vertex] + 1) {
                    atomicAdd(&sigma[v], sigma[curr_vertex]);
                }
            }
        }
    }
}


__global__ void dependencyKernel(VertexDictionary *d_vertex_dict, float *delta, int *sigma, int *dist, float *bc, int vertex_size, int source_vertex) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < vertex_size && dist[tid] != -1 && tid != source_vertex) {
        float delta_tid = delta[tid];

        int level = dist[tid];
        float sigma_tid = sigma[tid];

        unsigned long edge_blocks = d_vertex_dict->edge_block_count[tid];
        
        // Going to each neighbour of curr_vertex
        EdgeBlock *root = d_vertex_dict->edge_block_address[tid];
        EdgeBlock *curr;

        for(int j = 0 ; j < edge_blocks ; j++){
            unsigned long bit_string = bit_string_lookup[j];
            curr = traverse_bit_string(root, bit_string);

            for(int k = 0; k < EDGE_BLOCK_SIZE ; k++){
                unsigned long v = curr->edge_block_entry[k].destination_vertex;
                if(dist[v] == level+1 && sigma[v] > 0){
                    float contrib = ( sigma_tid / sigma[v]) * (1.0f + delta[v]);
                    delta_tid += contrib;
                }
            }
        }
 
        atomicAdd(&bc[tid], delta_tid);
    }
}
