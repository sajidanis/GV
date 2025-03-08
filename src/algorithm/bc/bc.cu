#include "bc.cuh"

void bc(GraphVine *graph, int source){

    auto &profiler = Profiler::getInstance();

    // Get the device vertex dictionary
    VertexDictionary *device_vertex_dictionary = graph->getVertexDictionary();

    auto vertex_size = graph->getVertexSize();

    thrust::device_vector<float> d_bc_vector(vertex_size, 0.0f);
    float *d_bc = thrust::raw_pointer_cast(d_bc_vector.data());

    thrust::device_vector<float> d_delta_vector(vertex_size, 0.0f);
    float *d_delta = thrust::raw_pointer_cast(d_delta_vector.data());
    
    thrust::device_vector<int> d_sigma_vector(vertex_size, 0);
    int *d_sigma = thrust::raw_pointer_cast(d_sigma_vector.data());

    thrust::device_vector<int> d_dist_vector(vertex_size, -1);
    int *d_dist = thrust::raw_pointer_cast(d_dist_vector.data());

    thrust::device_vector<int> d_frontier_vector(vertex_size, 0);
    int *d_frontier = thrust::raw_pointer_cast(d_frontier_vector.data());

    thrust::device_vector<int> d_next_frontier_vector(vertex_size, 0);
    int *d_next_frontier = thrust::raw_pointer_cast(d_next_frontier_vector.data());

    
    
    size_t thread_blocks = (vertex_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    int h_frontier_size = 1;
    int *d_frontier_size, *d_next_frontier_size;

    CUDA_CALL(cudaMalloc(&d_next_frontier_size, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_frontier_size, sizeof(int)));

    profiler.start("BC");

    for(int src = 0 ; src < vertex_size; src++){
        h_frontier_size = 1;

        CUDA_CALL(cudaMemset(d_next_frontier_size, 0, sizeof(int))); 

        // initialize bc arrays
        initializeArrays<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_sigma, d_delta, d_dist);
        CUDA_CALL(cudaDeviceSynchronize());

        // Set up the arrays for bfs
        initbfsKernel<<<1, 1>>>(src, d_sigma, d_dist, d_frontier, d_frontier_size);
        CUDA_CALL(cudaDeviceSynchronize());

        // Run the bfs
        while(h_frontier_size > 0){

            thread_blocks = (h_frontier_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
            bfsKernel<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, d_sigma, d_dist, d_frontier, d_frontier_size, d_next_frontier, d_next_frontier_size);
    
            CUDA_CALL(cudaDeviceSynchronize());
    
            std::swap(d_frontier_size, d_next_frontier_size);
    
            d_frontier_vector.swap(d_next_frontier_vector);
    
            d_frontier = thrust::raw_pointer_cast(d_frontier_vector.data());
            d_next_frontier = thrust::raw_pointer_cast(d_next_frontier_vector.data());
    
            CUDA_CALL(cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
    
            CUDA_CALL(cudaMemset(d_next_frontier_size, 0, sizeof(int)));
    
            CUDA_CALL(cudaDeviceSynchronize());
        }

        thread_blocks = (vertex_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        dependencyKernel<<<thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_dictionary, d_delta, d_sigma, d_dist, d_bc, vertex_size, src);
    }

    profiler.stop("BC");

    printDeviceVector("BC", d_bc_vector);

    CUDA_CALL(cudaFree(d_frontier_size));
    CUDA_CALL(cudaFree(d_next_frontier_size));
}