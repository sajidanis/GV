#include "csr.cuh"

GraphProperties *CSR::h_graph_prop = new GraphProperties();

// Read the graph from file
void CSR::readGraph(char * fileLocation) {
    auto &profiler = Profiler::getInstance();

    profiler.start("Read Graph");
    readFile(fileLocation, h_graph_prop, h_source, h_destination, h_source_degrees, 0);
    computeGraphProperties();
    initializeCSR();
    profiler.stop("Read Graph");

    std::cout << "[+] CSR initialization completed\n";
}

// Compute graph properties
void CSR::computeGraphProperties() {

    unsigned long vertex_size = h_graph_prop->xDim;
    unsigned long edge_size = h_graph_prop->total_edges;
    unsigned long batch_size = BATCH_SIZE;

    // Compute max and average degree
    h_graph_prop->max_degree = h_source_degrees[0];
    h_graph_prop->sum_degree = h_source_degrees[0];

    for (unsigned long i = 1; i < vertex_size; i++) {
        if (h_source_degrees[i] > h_graph_prop->max_degree) {
            h_graph_prop->max_degree = h_source_degrees[i];
        }
        h_graph_prop->sum_degree += h_source_degrees[i];
    }

    std::cout << "Max degree of the graph is " << h_graph_prop->max_degree << std::endl;
    std::cout << "Average degree of the graph is " << h_graph_prop->sum_degree / vertex_size << std::endl;

    // Compute batch information
    h_graph_prop->total_batches = static_cast<unsigned long>(std::ceil(static_cast<double>(edge_size) / batch_size));
    std::cout << "Batches required is " << h_graph_prop->total_batches << std::endl;

    h_source_degrees_new.resize(vertex_size);
    h_csr_offset_new.resize(vertex_size + 1);


    if (edge_size > batch_size) {
        h_csr_edges_new.resize(edge_size);
    } else {
        h_csr_edges_new.resize(batch_size);
    }
}

// Initialize CSR representation and resize data structures
void CSR::initializeCSR() {
    // create a csr based on the available source and destination data;
    unsigned long vertex_size = h_graph_prop->xDim;
    unsigned long edge_size = h_graph_prop->total_edges;

    h_csr_edges_new.resize(edge_size);

    thrust::exclusive_scan(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size + 1, h_csr_offset_new.begin());

    thrust::host_vector<unsigned long> index(vertex_size, 0);

    for (unsigned long i = 0; i < edge_size; ++i) {
        unsigned long src = h_source[i] - 1; // Convert to 0-index  
        unsigned long dest = h_destination[i]; // Destination node
        
        // Insert the destination node at the correct position
        unsigned long position = h_csr_offset_new[src] + index[src];
        h_csr_edges_new[position] = dest;

        // Update the insertion point for this source node
        index[src]++;
    }

}

void CSR::generate_random_batch(size_t batch_size, size_t batch_number){
    unsigned long vertex_size = h_graph_prop->xDim;
    h_graph_prop->batch_size = batch_size;
    thrust::fill(h_source_degrees.begin(), h_source_degrees.begin() + vertex_size, 0);

    h_source.resize(batch_size);
    h_destination.resize(batch_size);

    unsigned long seed = batch_number;
    unsigned long range = 0;
    unsigned long offset = 0;

    srand(seed + 1);
    for (unsigned long i = 0; i < batch_size / 2; ++i) {
        // EdgeUpdateType edge_update_data;
        unsigned long intermediate = rand() % ((range && (range < vertex_size)) ? range : vertex_size);
        unsigned long source;
        if (offset + intermediate < vertex_size)
            source = offset + intermediate;
        else
            source = intermediate;
        h_source[i] = source + 1;
        h_destination[i] = (rand() % vertex_size) + 1;
        h_source_degrees[source]++;
    }

    for (unsigned long i = batch_size / 2; i < batch_size; ++i) {
        // EdgeUpdateType edge_update_data;
        unsigned long intermediate = rand() % (vertex_size);
        unsigned long source;
        if (offset + intermediate < vertex_size)
            source = offset + intermediate;
        else
            source = intermediate;
        h_source[i] = source + 1;
        h_destination[i] = (rand() % vertex_size) + 1;
        h_source_degrees[source]++;
    }

    h_graph_prop->total_edges = batch_size;
    h_graph_prop->total_batches = static_cast<unsigned long>(std::ceil(static_cast<double>(batch_size) / BATCH_SIZE));

    computeGraphProperties();

    initializeCSR();   
}

thrust::host_vector<unsigned long> CSR::get_source_degree(){
    return h_source_degrees;
}

thrust::host_vector<unsigned long> CSR::get_csr_offset()
{
    return h_csr_offset_new;
}

thrust::host_vector<unsigned long> CSR::get_csr_edges()
{
    return h_csr_edges_new;
}

thrust::host_vector<unsigned long> CSR::get_source_degree_new(){
    return h_source_degrees_new;
}
