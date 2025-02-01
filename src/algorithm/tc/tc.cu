#include "tc.cuh"

void static_tc(GraphVine *graph, thrust::device_vector<float> &d_triangleCount) {

    auto &profiler = Profiler::getInstance();

    std::cout << "Page rank algorithm" << std::endl;

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    float *d_triangleCount_pointer = thrust::raw_pointer_cast(d_triangleCount.data());

    unsigned long *d_csr_offset_new_pointer = graph->getCsrOffsetNewPointer();
    unsigned long *d_csr_edges_new_pointer = graph->getCsrEdgesNewPointer();

    size_t vertex_size = graph->getVertexSize();
    size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);


    triangle_counting_kernel_VC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
    
    cudaDeviceSynchronize();

    printDeviceVector("Triangle Count", d_triangleCount);

}
