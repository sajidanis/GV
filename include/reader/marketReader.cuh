#ifndef MARKETREADER_CUH
#define MARKETREADER_CUH

#include <iostream>
#include <fstream>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <thrust/host_vector.h>
#include <cstdlib>


#include "graphStructures.cuh"
#include "util.cuh"

void readFile(char *fileLoc, GraphProperties *h_graph_prop, thrust::host_vector<unsigned long> &h_source, thrust::host_vector<unsigned long> &h_destination, thrust::host_vector<unsigned long> &h_source_degrees, unsigned long type);

#endif // MARKETREADER_CUH
