#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <chrono>

#include "marketReader.cuh"
#include "profiler.cuh"
#include "graphvine.cuh"
#include "csr.cuh"
#include "pagerank.cuh"
#include "tc.cuh"

int main(int argc, char **argv){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <market_file>" << " <GPU_ID>" << std::endl;
        return EXIT_FAILURE;
    }

    auto &profiler = Profiler::getInstance();

    char *fileLoc = argv[1];
    CSR *csr = new CSR();
    csr->readGraph(fileLoc);


    GraphVine *dynGraph = new GraphVine(csr);

    // Give options to run benchmarks and applications
    // Insertion type operations
    size_t kk = 0;

    size_t vertex_size = CSR::h_graph_prop->xDim;

    std::cout << "Starting Page Rank algorithm in main" << std::endl;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem << ", Total memory: " << total_mem << std::endl;
    std::cout << "Required memory: " << vertex_size * sizeof(float) * 2 << std::endl; // For two vectors

    thrust::device_vector<float> d_pageRankVector_1(vertex_size);
    thrust::device_vector<float> d_pageRankVector_2(vertex_size);

    // thrust::device_vector<float> d_triangleCount(vertex_size, 0);

    std::cout << "Starting Page Rank algorithm in main" << std::endl;

    // Run PR
    static_pagerank(dynGraph, 0.85, 0.0001, 100, d_pageRankVector_1, d_pageRankVector_2);

    // Run TC
    // static_tc(dynGraph, d_triangleCount);

    int choice;
    std::cout << "Enter type of insertion required" << std::endl
              << "1. Regular batched edge insertion" << std::endl
              << "2. Edge Insert and Delete performance benchmark" << std::endl
              << "3. Vertex Insert and Delete performance benchmark" << std::endl;
    std::cin >> choice;

    switch (choice) {
        case 1:
            // Regular batched edge insertion
            // dynGraph->bulkBuild();
            break;
        case 2:
            // Edge Insert and Delete performance benchmark
            profiler.start("Generate Random Batch");
            csr->generate_random_batch(BATCH_SIZE, kk);
            profiler.stop("Generate Random Batch");

            dynGraph->batchInsert(csr, kk);

            dynamic_pagerank(dynGraph, 0.85, 0.0001, 100, d_pageRankVector_1, d_pageRankVector_2);

            static_pagerank(dynGraph, 0.85, 0.0001, 100, d_pageRankVector_1, d_pageRankVector_2);

            break;
        case 3:
            // Vertex Insert and Delete performance benchmark
            break;
        default:
            std::cerr << "Invalid choice" << std::endl;
            break;
    }

    // Run Page rank algorithm
    

    return 0;
}