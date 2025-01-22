#ifndef CSR_CUH
#define CSR_CUH

#include <thrust/host_vector.h>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "marketReader.cuh"
#include "graphStructures.cuh"
#include "util.cuh"
#include "profiler.cuh"

class CSR{
public:
    CSR() = default;
    ~CSR() = default;

    void readGraph(char *fileLocation);
    void computeGraphProperties();
    void initializeCSR();

    void generate_random_batch(size_t batch_size, size_t batch_number);

    static GraphProperties* h_graph_prop;

    thrust::host_vector<unsigned long> get_source_degree();
    thrust::host_vector<unsigned long> get_source_degree_new();
    thrust::host_vector<unsigned long> get_csr_offset();
    thrust::host_vector<unsigned long> get_csr_edges();

private:

    // Private members
    char *fileLocation;
    thrust::host_vector<unsigned long> h_source;
    thrust::host_vector<unsigned long> h_destination;
    thrust::host_vector<unsigned long> h_source_degrees;
    thrust::host_vector<unsigned long> h_source_degrees_new;
    thrust::host_vector<unsigned long> h_csr_offset_new;
    thrust::host_vector<unsigned long> h_csr_edges_new;

    // // Private functions
    // unsigned long calculatePowerBase2(unsigned long exponent);
    // unsigned long calculateLogBase2(unsigned long value);

};


#endif