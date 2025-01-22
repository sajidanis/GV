#include "graphvine.cuh"

__device__ EdgePreallocatedQueue d_e_queue;

GraphVine::GraphVine(CSR *csr) : h_csr(csr) {
    auto &profiler = Profiler::getInstance();

    profiler.logMemoryUsage();

    profiler.start("Initiate Vertex Dictionary and Edge Blocks");
    initiateVertexDictionary();
    profiler.stop("Initiate Vertex Dictionary and Edge Blocks");

    profiler.start("Initiate Device Vectors");
    initiateDeviceVectors();
    profiler.stop("Initiate Device Vectors");

    profiler.start("Copy Graph Data From Host To Device");
    copyGraphDataFromHostToDevice();
    profiler.stop("Copy Graph Data From Host To Device");

    profiler.logMemoryUsage();

    profiler.start("Bulk Build");
    bulkBuild();
    profiler.stop("Bulk Build");

    profiler.logMemoryUsage();

}

// Function to initialize GPU memory for the vertex dictionary and related data
void GraphVine::initiateVertexDictionary() {
    // Create the vertex dictionary size based on the number of vertices and take it as a nearest power of two
    size_t vertexSize = CSR::h_graph_prop->xDim;

    vertexDictionarySize = computeNearestPowerOf2(vertexSize);

    std::cout << "Vertex Dictionary size: " << vertexDictionarySize << std::endl;

    // Vertex dictionary structure
    cudaMalloc(&d_vertexDictionary, sizeof(VertexDictionary));

    // GPU memory allocation for vertex dictionary arrays
    unsigned long *d_vertex_id, *d_edge_block_count_VD, *d_last_insert_edge_offset;
    unsigned int *d_active_edge_count_VD;
    
    EdgeBlock **d_last_insert_edge_block, **d_edge_block_address;

    cudaMalloc(&d_vertex_id, 3 * vertexDictionarySize * sizeof(unsigned long)); // Allocating all the memories in single call
    
    d_edge_block_count_VD = d_vertex_id + vertexDictionarySize; // from the earlier allocation just manipulating pointer
    d_last_insert_edge_offset = d_vertex_id + (2 * vertexDictionarySize);

    cudaMalloc(&d_active_edge_count_VD, vertexDictionarySize * sizeof(unsigned int));

    cudaMalloc(reinterpret_cast<void**>(&d_last_insert_edge_block), 2 * vertexDictionarySize * sizeof(EdgeBlock *));
   
    d_edge_block_address = d_last_insert_edge_block + vertexDictionarySize;
    
    // Ensure all allocations are synchronized
    cudaDeviceSynchronize();

    std::cout << "All the structures related to vertex Dictionary has been initialised." << std::endl;

    // Initiating edge block queues 
    initiateEdgeBlocks();

    std::cout << "Edge Blocks has been initialized" << std::endl;

    // Call the kernel to initiate the vertex dictionary
    data_structure_init<<<1, 1>>>(d_vertexDictionary, d_vertex_id, d_edge_block_count_VD, d_active_edge_count_VD, d_last_insert_edge_offset, d_last_insert_edge_block, d_edge_block_address, d_queue_edge_block_address);


    // push the edgeblock to the device and set up the queue

    unsigned long thread_blocks = ceil(double(edge_block_count_device) / THREADS_PER_BLOCK);

    parallel_push_edge_preallocate_list_to_device_queue<<<thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, edge_block_count_device, edge_block_count_device);

    parallel_push_queue_update<<<1, 1>>>(edge_block_count_device);

    // we need to initiate each vertex id with vertexDictionary internal mapping
    thread_blocks = ceil(double(vertexSize) / THREADS_PER_BLOCK);

    parallel_vertex_dictionary_init_v1<<<thread_blocks, THREADS_PER_BLOCK>>>(vertexSize, d_vertexDictionary);

    cudaDeviceSynchronize();
}

void GraphVine::initiateEdgeBlocks(){
    size_t vertexSize = CSR::h_graph_prop->xDim;
    h_edge_blocks_count_init.resize(vertexSize);
    unsigned long total_edge_blocks_count_init = 0;
    // std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;

    auto source_degrees = h_csr->get_source_degree();

    for (unsigned long i = 0; i < vertexSize; i++){
        // unsigned long edge_blocks = ceil(double(source_degrees[i]) / EDGE_BLOCK_SIZE);
        unsigned long edge_blocks = (source_degrees[i] + EDGE_BLOCK_SIZE) / EDGE_BLOCK_SIZE;
        h_edge_blocks_count_init[i] = edge_blocks;
        total_edge_blocks_count_init += edge_blocks;
    }

    total_edge_blocks_count_init = total_edge_blocks_count_init * 10;

    printf("Total edge blocks needed = %lu, %luMB\n", total_edge_blocks_count_init, (total_edge_blocks_count_init * sizeof(EdgeBlock)) / 1024 / 1024);

    edge_block_count_device = get_edge_block_count_device();
    cudaDeviceSynchronize();

    // cudaMalloc((struct edge_block**)&device_edge_block, total_edge_blocks_count_init * sizeof(struct edge_block));
    cudaMalloc((struct edge_block **)&device_edge_block, edge_block_count_device * sizeof(EdgeBlock));
    cudaMalloc((struct edge_block **)&d_queue_edge_block_address, edge_block_count_device * sizeof(EdgeBlock *));

    // cudaMalloc((struct edge_block**)&device_edge_block, 2 * total_edge_blocks_count_init * sizeof(struct edge_block));
    cudaDeviceSynchronize();
}

void GraphVine::initiateDeviceVectors(){
    size_t edge_size = CSR::h_graph_prop->total_edges;
    size_t vertex_size = CSR::h_graph_prop->xDim;

    d_csr_edges_new = thrust::device_vector<unsigned long>(edge_size);
    d_source_degrees_new = thrust::device_vector<unsigned long>(vertex_size);
    d_csr_offset_new = thrust::device_vector<unsigned long> (vertex_size + 1);

    d_edge_blocks_count = thrust::device_vector<unsigned long>(vertex_size + 1);
    d_prefix_sum_edge_blocks_new = thrust::device_vector<unsigned long> (vertex_size + 1);

    d_source_vector = thrust::device_vector<unsigned long>(vertex_size + 1);
    d_source_vector_1 = thrust::device_vector<unsigned long> (1);
    d_thread_count_vector = thrust::device_vector<unsigned long> (vertex_size + 1);


    // Pointer cast for GPU handling
    d_source_degrees_new_pointer = thrust::raw_pointer_cast(d_source_degrees_new.data());
    d_csr_offset_new_pointer = thrust::raw_pointer_cast(d_csr_offset_new.data());
    d_csr_edges_new_pointer = thrust::raw_pointer_cast(d_csr_edges_new.data());
    d_edge_blocks_count_pointer = thrust::raw_pointer_cast(d_edge_blocks_count.data());
    d_prefix_sum_edge_blocks_new_pointer = thrust::raw_pointer_cast(d_prefix_sum_edge_blocks_new.data());

    d_source_vector_pointer = thrust::raw_pointer_cast(d_source_vector.data());
    d_source_vector_1_pointer = thrust::raw_pointer_cast(d_source_vector_1.data());
}

void GraphVine::copyGraphDataFromHostToDevice(){
    auto h_csr_offset_new = h_csr->get_csr_offset();
    auto h_csr_edges_new = h_csr->get_csr_edges();
    auto h_source_degrees = h_csr->get_source_degree();

    thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
    thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
    thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new.begin());
    cudaDeviceSynchronize();
}

void GraphVine::bulkBuild(){
    std::cout << "Bulk build real graph now" << std::endl;
    unsigned long insert_load_factor_VC = 2;

    size_t edge_size = CSR::h_graph_prop->total_edges;
    size_t vertex_size = CSR::h_graph_prop->xDim;

    unsigned long start_index = 0;
    unsigned long end_index = edge_size;

    unsigned long remaining_edges = edge_size - start_index;

    unsigned long current_batch = end_index - start_index;

    size_t thread_blocks = ceil(double(edge_size) / THREADS_PER_BLOCK);


    cudaMemcpy(&total_edge_blocks_count_batch, d_prefix_sum_edge_blocks_new_pointer + vertex_size, sizeof(unsigned long), cudaMemcpyDeviceToHost);

    batched_edge_inserts_EC<<<thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, d_vertexDictionary, 0, edge_size, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

    batched_edge_inserts_EC_postprocessing<<<thread_blocks, THREADS_PER_BLOCK>>>(d_vertexDictionary, vertex_size, edge_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer, d_prefix_sum_edge_blocks_new_pointer, 0);

    update_edge_queue<<<1, 1>>>(total_edge_blocks_count_batch);

    cudaDeviceSynchronize();
}

void GraphVine::batchInsert(CSR *csr) {

    auto &profiler = Profiler::getInstance();

    auto h_csr_offset_new = csr->get_csr_offset();
    auto h_csr_edges_new = csr->get_csr_edges();
    auto h_source_degrees = csr->get_source_degree();

    // auto h_source_degrees_new = csr->get_source_degree_new();

    // Ensure device vectors are correctly sized
    if (d_csr_offset_new.size() != h_csr_offset_new.size()) {
        d_csr_offset_new.resize(h_csr_offset_new.size());
        d_csr_offset_new_pointer = thrust::raw_pointer_cast(d_csr_offset_new.data());
    }
    if (d_csr_edges_new.size() != h_csr_edges_new.size()) {
        d_csr_edges_new.resize(h_csr_edges_new.size());
        d_csr_edges_new_pointer = thrust::raw_pointer_cast(d_csr_edges_new.data());
    }
    if (d_source_degrees_new.size() != h_source_degrees.size()) {
        d_source_degrees_new.resize(h_source_degrees.size());
        d_source_degrees_new_pointer = thrust::raw_pointer_cast(d_source_degrees_new.data());
        // h_source_degrees_new.resize(h_source_degrees.size());
    }

    auto vertex_size = csr->h_graph_prop->xDim;
    auto batch_size = csr->h_graph_prop->batch_size;

    thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
    thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
    thrust::copy(h_source_degrees.begin(), h_source_degrees.end(), d_source_degrees_new.begin());

    cudaDeviceSynchronize();

    // Batch Duplicates Removal

    profiler.start("BATCH DUPLICATION");

    unsigned long thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);

    device_remove_batch_duplicates<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, batch_size,d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

    cudaDeviceSynchronize();
    std::cout << "Removed intra-batch duplicates" << std::endl;

    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);

    device_update_source_degrees<<<thread_blocks, THREADS_PER_BLOCK>>>(vertex_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_source_degrees_new_pointer);

    cudaDeviceSynchronize();

    std::cout << "Source Degree Updated\n";

    thrust::host_vector<unsigned long> h_source_degrees_new = d_source_degrees_new;

    cudaDeviceSynchronize();

    auto max_degree_batch = h_source_degrees_new[0];
    auto sum_degree_batch = h_source_degrees_new[0];

    for (unsigned long j = 1; j < vertex_size; j++) {
        if (h_source_degrees_new[j] > max_degree_batch)
            max_degree_batch = h_source_degrees_new[j];
        sum_degree_batch += h_source_degrees_new[j];
    }

    auto average_degree_batch = sum_degree_batch / vertex_size;

    std::cout << "Max degree of batch is " << max_degree_batch << std::endl;
    std::cout << "Average degree of batch is " << sum_degree_batch / vertex_size << std::endl;

    profiler.stop("BATCH DUPLICATION");


    if ((EDGE_BLOCK_SIZE > 40) && (BATCH_SIZE <= 10000000)) {

        unsigned long thread_count_EBC;
        thrust::fill(d_source_vector.begin(), d_source_vector.end(), 0);

        thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    
        batched_delete_preprocessing_EC_LD<<<thread_blocks, THREADS_PER_BLOCK>>>(d_vertexDictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer, d_source_vector_pointer);

        cudaDeviceSynchronize();

        thrust::exclusive_scan(d_source_vector.begin(), d_source_vector.begin() + vertex_size + 1, d_source_vector.begin());

        cudaDeviceSynchronize();

        cudaMemcpy(&thread_count_EBC, d_source_vector_pointer + vertex_size, sizeof(unsigned long), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
      
        thread_blocks = ceil(double(thread_count_EBC) / THREADS_PER_BLOCK);

        batched_delete_kernel_EC_LD<<<thread_blocks, THREADS_PER_BLOCK>>>(d_vertexDictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
    } else {
        thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);

        batched_delete_kernel_EC_HD<<<thread_blocks, THREADS_PER_BLOCK>>>(d_vertexDictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_vector_pointer);
    }


}

unsigned long GraphVine::get_edge_block_count_device(){

    // show memory usage of GPU

    size_t free_byte;

    size_t total_byte;

    cudaMemGetInfo(&free_byte, &total_byte);

    double free_db = (double)free_byte;

    double total_db = (double)total_byte;

    double used_db = total_db - free_db;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

           used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

    unsigned long half_device_memory = total_db / 2;
    unsigned long edge_blocks_possible = half_device_memory / (sizeof(EdgeBlock));
    printf("%luB, %lu edge_blocks\n", half_device_memory, edge_blocks_possible);

    return edge_blocks_possible;
}


// 