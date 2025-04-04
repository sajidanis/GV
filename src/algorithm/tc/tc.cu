#include "tc.cuh"

void static_tc(GraphVine *graph, thrust::device_vector<float> &d_triangleCount) {

    auto &profiler = Profiler::getInstance();

    std::cout << "Page rank algorithm" << std::endl;

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    float *d_triangleCount_pointer = thrust::raw_pointer_cast(d_triangleCount.data());

    size_t vertex_size = graph->getVertexSize();
    size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);

    profiler.start("TC");
  
    tc_kernel_VC_sorted<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer);

    cudaDeviceSynchronize();

    profiler.stop("TC");

    printDeviceVector("Triangle Count", d_triangleCount);

}

void dynamic_tc(GraphVine *graph, thrust::device_vector<float> &d_triangleCount){
    auto &profiler = Profiler::getInstance();

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    float *d_triangleCount_pointer = thrust::raw_pointer_cast(d_triangleCount.data());

    unsigned long *d_csr_offset_new_pointer = graph->getCsrOffsetNewPointer();
    unsigned long *d_csr_edges_new_pointer = graph->getCsrEdgesNewPointer();

    unsigned long *d_affected_nodes_pointer = graph->getAffectedNodesPointer();

    size_t vertex_size = graph->getVertexSize();
    size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);


    profiler.start("Dynamic Triangle Count");

    std::cout << "Starting dynamic triangle count algorithm" << std::endl;

    dynamic_tc_kernel_VC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_triangleCount_pointer, d_affected_nodes_pointer);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    profiler.stop("Dynamic Triangle Count");

    std::cout << "Stopping dynamic triangle count algorithm" << std::endl;

    printDeviceVector("Triangle Count", d_triangleCount);
}
