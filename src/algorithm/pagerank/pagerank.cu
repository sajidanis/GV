#include "pagerank.cuh"

void static_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter) {

    auto &profiler = Profiler::getInstance();

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    // Get the source vector
    unsigned long *d_source_vector_pointer = graph->getSourceVectorPointer();

    auto vertex_size = graph->getVertexSize();

    thrust::device_vector<float> d_pageRankVector_1(vertex_size);
    thrust::device_vector<float> d_pageRankVector_2(vertex_size);

    float *d_pageRankVector_1_pointer = thrust::raw_pointer_cast(d_pageRankVector_1.data());
    float *d_pageRankVector_2_pointer = thrust::raw_pointer_cast(d_pageRankVector_2.data());

    profiler.start("Page Rank");

    thrust::fill(d_pageRankVector_1.begin(), d_pageRankVector_1.end(), 0.25f);

    size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    pageRank_kernel_HD<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_pageRankVector_1_pointer, d_pageRankVector_2_pointer, d_source_vector_pointer);

    cudaDeviceSynchronize();

    profiler.stop("Page Rank");

    printDeviceVector("Page Rank", d_pageRankVector_2);

}