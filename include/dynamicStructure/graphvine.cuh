#ifndef GRAPHVINE_CUH
#define GRAPHVINE_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <numeric>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include "marketReader.cuh"
#include "graphStructures.cuh"
#include "constants.cuh"
#include "util.cuh"
#include "csr.cuh"
#include "profiler.cuh"

#include "init_kernels.cuh"
#include "insert_kernels.cuh"

// Global Variables
extern __device__ EdgePreallocatedQueue d_e_queue;

class GraphVine{
public:
    GraphVine(CSR *csr);
    ~GraphVine() = default;
    void initiateVertexDictionary();
    void initiateEdgeBlocks();
    void initiateDeviceVectors();

    void bulkBuild();

    void batchInsert(CSR *csr, size_t kk);

    void copyGraphDataFromHostToDevice();

    VertexDictionary* getVertexDictionary();
    unsigned long* getSourceVectorPointer();
    unsigned long getVertexSize();

private:
    
    // Private members for Vertex Dictionary
    VertexDictionary *d_vertexDictionary;
    size_t vertexDictionarySize;
    CSR *h_csr;

    // Private members for Edge Blocks
    EdgeBlock *device_edge_block;
    EdgeBlock **d_queue_edge_block_address;
    thrust::host_vector<unsigned long> h_edge_blocks_count_init;
    unsigned long edge_block_count_device;
    
    unsigned long get_edge_block_count_device();

    //Device Vectors
    thrust::device_vector<unsigned long> d_csr_edges_new;

    thrust::device_vector<unsigned long> d_source_degrees_new;
    thrust::device_vector<unsigned long> d_csr_offset_new;
   
    thrust::device_vector<unsigned long> d_edge_blocks_count;
    thrust::device_vector<unsigned long> d_prefix_sum_edge_blocks_new;

    thrust::device_vector<unsigned long> d_source_vector;
    thrust::device_vector<unsigned long> d_source_vector_1;
    thrust::device_vector<unsigned long> d_thread_count_vector;

    thrust::device_vector<unsigned long> d_affected_nodes;

    unsigned long total_edge_blocks_count_batch;

    unsigned long *d_source_degrees_new_pointer;
    unsigned long *d_csr_offset_new_pointer;
    unsigned long *d_csr_edges_new_pointer;
    unsigned long *d_edge_blocks_count_pointer;
    unsigned long *d_prefix_sum_edge_blocks_new_pointer;

    unsigned long *d_source_vector_pointer;
    unsigned long *d_source_vector_1_pointer;

    unsigned long *d_affected_nodes_pointer;
};


#endif //GRAPHVINE_CUH