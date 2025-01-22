#include "util.cuh"

template<typename T>
void printHostVector(const std::string name, const thrust::host_vector<T> &vec){
    int i = 0;
    std::cout <<"[" << name << "] -> ";
    for(auto &el : vec){
        std::cout << el << " ";
        i++;
        if(i > 40) break;
    }
    std::cout << "\n";
}

__global__ void printDeviceVectorKernel(const unsigned long* d_vec, size_t size) {
    for(int i = 0 ; i < size; i++){
        printf(" %lu ", d_vec[i]);
    }
}

template<typename T>
void printDeviceVector(const std::string name, const thrust::device_vector<T>& d_vec) {
    size_t size = d_vec.size() >= 40 ? 40 : d_vec.size();
    std::cout <<"[" << name << "] -> ";
    // Launch kernel to print device vector
    const unsigned long* d_vec_ptr = thrust::raw_pointer_cast(d_vec.data());

    // Call the kernel
    printDeviceVectorKernel<<<1, 1>>>(d_vec_ptr, size);

    // Synchronize to ensure all prints are complete
    cudaDeviceSynchronize();

    std::cout << "\n";
}

template void printHostVector<unsigned long>(const std::string name, const thrust::host_vector<unsigned long> &vec);
template void printDeviceVector<unsigned long>(const std::string name, const thrust::device_vector<unsigned long> &d_vec);

unsigned long computeNearestPowerOf2(unsigned long num){
    if (num == 0) return 1; // Handle the case where n is 0, since 0 doesn't have a meaningful power of two ceiling.

    // If n is already a power of two, return n.
    if ((num & (num - 1)) == 0) return num;

    // Use the bit trick to calculate the next power of two.
    num--;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    num |= num >> 32;
    return num + 1;
}


// Function to print the Edge structure
// void printEdge(const Edge &edge) {
//     std::cout << "Destination Vertex: " << edge.destination_vertex << std::endl;
// }

// // Function to print the EdgeBlock structure
// void printEdgeBlock(const EdgeBlock *h_edge_block, unsigned long block_size) {
//     std::cout << "EdgeBlock:" << std::endl;
//     std::cout << "  Active Edge Count: " << h_edge_block->active_edge_count << std::endl;

//     for (unsigned short i = 0; i < h_edge_block->active_edge_count; ++i) {
//         std::cout << "  Edge[" << i << "]: ";
//         printEdge(h_edge_block->edge_block_entry[i]);
//     }

//     std::cout << "  Left Pointer: " << h_edge_block->lptr << std::endl;
//     std::cout << "  Right Pointer: " << h_edge_block->rptr << std::endl;
//     std::cout << "  Level Order Predecessor: " << h_edge_block->level_order_predecessor << std::endl;
// }

// // Function to print the VertexDictionary structure
// void printVertexDictionary(const VertexDictionary &h_vertex_dict, unsigned long vertex_count) {
//     std::cout << "Vertex Dictionary:" << std::endl;

//     for (unsigned long i = 0; i < vertex_count; ++i) {
//         std::cout << "Vertex[" << i << "]:" << std::endl;
//         std::cout << "  Vertex ID: " << h_vertex_dict.vertex_id[i] << std::endl;
//         std::cout << "  Edge Block Count: " << h_vertex_dict.edge_block_count[i] << std::endl;
//         std::cout << "  Active Edge Count: " << h_vertex_dict.active_edge_count[i] << std::endl;
//         std::cout << "  Last Insert Edge Offset: " << h_vertex_dict.last_insert_edge_offset[i] << std::endl;
//         std::cout << "  Last Insert Edge Block: " << h_vertex_dict.last_insert_edge_block[i] << std::endl;
//     }
// }

// // Function to print the EdgePreallocatedQueue structure
// void printEdgePreallocatedQueue(const EdgePreallocatedQueue &h_queue, unsigned long queue_size) {
//     std::cout << "Edge Preallocated Queue:" << std::endl;
//     std::cout << "  Front: " << h_queue.front << std::endl;
//     std::cout << "  Rear: " << h_queue.rear << std::endl;
//     std::cout << "  Count: " << h_queue.count << std::endl;

//     for (long i = 0; i < h_queue.count; ++i) {
//         long index = (h_queue.front + i) % queue_size;
//         std::cout << "  EdgeBlock Address[" << index << "]: " << h_queue.edge_block_address[index] << std::endl;
//     }
// }

// // Main function to copy data from device to host and visualize
// void visualizeGraph(const VertexDictionary *d_vertex_dict, 
//                     const EdgeBlock *d_edge_blocks, 
//                     const EdgePreallocatedQueue *d_queue, 
//                     unsigned long vertex_count, 
//                     unsigned long edge_block_count, 
//                     unsigned long queue_size) {

//     // Host copies
//     VertexDictionary h_vertex_dict;
//     EdgePreallocatedQueue h_queue;

//     // Allocate host memory for VertexDictionary arrays
//     h_vertex_dict.vertex_id = new unsigned long[vertex_count];
//     h_vertex_dict.edge_block_count = new unsigned long[vertex_count];
//     h_vertex_dict.active_edge_count = new unsigned int[vertex_count];
//     h_vertex_dict.last_insert_edge_offset = new unsigned long[vertex_count];
//     h_vertex_dict.last_insert_edge_block = new EdgeBlock*[vertex_count];

//     // Copy VertexDictionary from device to host
//     cudaMemcpy(h_vertex_dict.vertex_id, d_vertex_dict->vertex_id, vertex_count * sizeof(unsigned long), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_vertex_dict.edge_block_count, d_vertex_dict->edge_block_count, vertex_count * sizeof(unsigned long), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_vertex_dict.active_edge_count, d_vertex_dict->active_edge_count, vertex_count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_vertex_dict.last_insert_edge_offset, d_vertex_dict->last_insert_edge_offset, vertex_count * sizeof(unsigned long), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_vertex_dict.last_insert_edge_block, d_vertex_dict->last_insert_edge_block, vertex_count * sizeof(EdgeBlock*), cudaMemcpyDeviceToHost);

//     // Allocate host memory for EdgeBlock array
//     EdgeBlock *h_edge_blocks = new EdgeBlock[edge_block_count];

//     // Copy EdgeBlocks from device to host
//     cudaMemcpy(h_edge_blocks, d_edge_blocks, edge_block_count * sizeof(EdgeBlock), cudaMemcpyDeviceToHost);

//     // Copy EdgePreallocatedQueue from device to host
//     cudaMemcpy(&h_queue, d_queue, sizeof(EdgePreallocatedQueue), cudaMemcpyDeviceToHost);

//     // Allocate host memory for queue edge block addresses
//     h_queue.edge_block_address = new EdgeBlock*[queue_size];
//     cudaMemcpy(h_queue.edge_block_address, d_queue->edge_block_address, queue_size * sizeof(EdgeBlock*), cudaMemcpyDeviceToHost);

//     // Print the structures
//     std::cout << "========== Graph Visualization ==========" << std::endl;

//     printVertexDictionary(h_vertex_dict, vertex_count);

//     for (unsigned long i = 0; i < edge_block_count; ++i) {
//         std::cout << "\nEdgeBlock[" << i << "]:" << std::endl;
//         printEdgeBlock(&h_edge_blocks[i], EDGE_BLOCK_SIZE);
//     }

//     printEdgePreallocatedQueue(h_queue, queue_size);

//     // Cleanup host memory
//     delete[] h_vertex_dict.vertex_id;
//     delete[] h_vertex_dict.edge_block_count;
//     delete[] h_vertex_dict.active_edge_count;
//     delete[] h_vertex_dict.last_insert_edge_offset;
//     delete[] h_vertex_dict.last_insert_edge_block;

//     delete[] h_edge_blocks;
//     delete[] h_queue.edge_block_address;
// }