#include "pagerank.cuh"

void static_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter, thrust::device_vector<float> &d_pageRankVector_1, thrust::device_vector<float> &d_pageRankVector_2) {

    auto &profiler = Profiler::getInstance();

    std::cout << "Page rank algorithm" << std::endl;

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    auto vertex_size = graph->getVertexSize();

    float *d_prev_pr_ptr = thrust::raw_pointer_cast(d_pageRankVector_1.data());
    float *d_curr_pr_ptr = thrust::raw_pointer_cast(d_pageRankVector_2.data());

    thrust::fill(d_pageRankVector_1.begin(), d_pageRankVector_1.end(), 1.0f / vertex_size);

    thrust::device_vector<float> d_difference_vector(vertex_size, 0.0f);
    float *d_difference_ptr = thrust::raw_pointer_cast(d_difference_vector.data());

    float normalized_damp = (1.0f - alpha) / vertex_size;
    
    float h_reduction = 0.0f;
    float *d_reduction;

    CUDA_CALL(cudaMalloc(&d_reduction, sizeof(float)));
    

    profiler.start("Page Rank");

    int iter = 0;

    while(iter < max_iter) {
        CUDA_CALL(cudaMemset(d_reduction, 0, sizeof(float)));
        
        size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);

        pageRank_kernel_HD<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_prev_pr_ptr, d_curr_pr_ptr);

        CUDA_KERNEL_CHECK();
        cudaDeviceSynchronize();

        pagerank_post_kernel<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, alpha, normalized_damp, d_prev_pr_ptr, d_curr_pr_ptr, d_difference_ptr);

        h_reduction = thrust::reduce(d_difference_vector.begin(), d_difference_vector.end(), 0.0f, thrust::plus<float>());
        // reduceSum(d_difference_ptr, d_reduction, vertex_size);

        CUDA_KERNEL_CHECK();
        cudaDeviceSynchronize();

        // cudaMemcpy(&h_reduction, d_reduction, sizeof(float), cudaMemcpyDeviceToHost);

        // cudaDeviceSynchronize();

        if(h_reduction < epsilon)
            break;

        iter++;
    }

    profiler.stop("Page Rank");

    printDeviceVector("Page Rank", d_pageRankVector_1);

    cudaFree(d_reduction);

}

void dynamic_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter, thrust::device_vector<float> &d_pageRankVector_1, thrust::device_vector<float> &d_pageRankVector_2) {
    auto &profiler = Profiler::getInstance();

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    // Get the Affected Nodes
    unsigned long *d_affected_nodes_pointer = graph->getAffectedNodesPointer();

    auto vertex_size = graph->getVertexSize();

    float *d_prev_pr_ptr = thrust::raw_pointer_cast(d_pageRankVector_1.data());
    float *d_curr_pr_ptr = thrust::raw_pointer_cast(d_pageRankVector_2.data());

    thrust::device_vector<float> d_difference_vector(vertex_size, 0.0f);
    float *d_difference_ptr = thrust::raw_pointer_cast(d_difference_vector.data());

    float normalized_damp = (1.0f - alpha) / vertex_size;
    
    float h_reduction = 0.0f;
    float *d_reduction_dyn;

    CUDA_CALL(cudaMalloc(&d_reduction_dyn, sizeof(float)));

    profiler.start("Dynamic Page Rank");

    int iter = 0;

    std::cout << "Starting Page Rank algorithm" << std::endl;

    while(iter < max_iter) {
        CUDA_CALL(cudaMemset(d_reduction_dyn, 0, sizeof(float)));
        
        size_t thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);

        dynamic_pageRank_kernel_HD<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, d_affected_nodes_pointer, d_prev_pr_ptr, d_curr_pr_ptr);

        CUDA_KERNEL_CHECK();
        cudaDeviceSynchronize();

        pagerank_post_kernel_dynamic<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, vertex_size, alpha, normalized_damp, d_prev_pr_ptr, d_curr_pr_ptr, d_difference_ptr, d_affected_nodes_pointer);

        h_reduction = thrust::reduce(d_difference_vector.begin(), d_difference_vector.end(), 0.0f, thrust::plus<float>());

        CUDA_KERNEL_CHECK();
        // cudaDeviceSynchronize();

        // cudaMemcpy(&h_reduction, d_reduction_dyn, sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        if(h_reduction < epsilon)
            break;

        iter++;
    }

    profiler.stop("Dynamic Page Rank");

    printDeviceVector("Page Rank", d_pageRankVector_1);

    cudaFree(d_reduction_dyn);
}
