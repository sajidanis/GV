#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <chrono>

#include "marketReader.cuh"
#include "profiler.cuh"
#include "graphvine.cuh"
#include "csr.cuh"


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
    int choice;
    std::cout << "Enter type of insertion required" << std::endl
              << "1. Regular batched edge insertion" << std::endl
              << "2. Edge Insert and Delete performance benchmark" << std::endl
              << "3. Vertex Insert and Delete performance benchmark" << std::endl;
    std::cin >> choice;

    size_t kk = 0;

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

            printHostVector("CSR Offset of Batch Data", csr->get_csr_offset());
            printHostVector("CSR Edges of Batch Data", csr->get_csr_edges());
            printHostVector("Source Degrees", csr->get_source_degree());

            dynGraph->batchInsert(csr);

            break;
        case 3:
            // Vertex Insert and Delete performance benchmark
            break;
        default:
            std::cerr << "Invalid choice" << std::endl;
            break;
    }

    return 0;
}