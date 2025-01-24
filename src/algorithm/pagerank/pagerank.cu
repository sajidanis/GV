#include "pagerank.cuh"

void static_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter, thrust::device_vector<float> &d_pageRankVector_1, thrust::device_vector<float> &d_pageRankVector_2) {

    auto &profiler = Profiler::getInstance();

    std::cout << "Page rank algorithm" << std::endl;

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    // Get the source vector
    unsigned long *d_source_vector_pointer = graph->getSourceVectorPointer();

    auto vertex_size = graph->getVertexSize();

    float *d_pageRankVector_1_pointer = thrust::raw_pointer_cast(d_pageRankVector_1.data());
    float *d_pageRankVector_2_pointer = thrust::raw_pointer_cast(d_pageRankVector_2.data());

    profiler.start("Page Rank");

    std::cout << "Starting page rank algorithm" << std::endl;

    thrust::fill(d_pageRankVector_1.begin(), d_pageRankVector_1.end(), 0.25f);

    size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    pageRank_kernel_HD<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_pageRankVector_1_pointer, d_pageRankVector_2_pointer, d_source_vector_pointer);

    cudaDeviceSynchronize();

    profiler.stop("Page Rank");

    printDeviceVector("Page Rank", d_pageRankVector_2);

}

void dynamic_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter, thrust::device_vector<float> &d_pageRankVector_1, thrust::device_vector<float> &d_pageRankVector_2) {
    auto &profiler = Profiler::getInstance();

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    // Get the source vector
    unsigned long *d_source_vector_pointer = graph->getSourceVectorPointer();

    // Get the Affected Nodes
    unsigned long *d_affected_nodes_pointer = graph->getAffectedNodesPointer();

    auto vertex_size = graph->getVertexSize();

    float *d_pageRankVector_1_pointer = thrust::raw_pointer_cast(d_pageRankVector_1.data());
    float *d_pageRankVector_2_pointer = thrust::raw_pointer_cast(d_pageRankVector_2.data());

    profiler.start("Dynamic Page Rank");

    size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    dynamic_pageRank_kernel_HD<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size,  d_affected_nodes_pointer, d_pageRankVector_1_pointer, d_pageRankVector_2_pointer);

    cudaDeviceSynchronize();

    profiler.stop("Dynamic Page Rank");

    printDeviceVector("Page Rank", d_pageRankVector_2);
}
