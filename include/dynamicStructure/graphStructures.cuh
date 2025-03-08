#ifndef GRAPH_STRUCTURES_CUH
#define GRAPH_STRUCTURES_CUH

#include "constants.cuh"



struct GraphProperties{
    unsigned long xDim;
    unsigned long yDim;
    unsigned long total_edges;

    unsigned long max_degree;
    unsigned long sum_degree;
    unsigned long total_batches;
    unsigned long batch_size = BATCH_SIZE;
};

struct Edge {
    unsigned long destination_vertex;
    // unsigned long weight;
    // unsigned long timestamp;
};

struct EdgeBlock {
    Edge edge_block_entry[EDGE_BLOCK_SIZE]; // instead of an array we need to change it to hashmap kind of structure
    unsigned int active_edge_count;
    EdgeBlock *lptr;
    EdgeBlock *rptr;
    EdgeBlock *level_order_predecessor;
    unsigned long *src_vertex;
};

/** 
// struct EdgeBlock {
//     int keys[HASH_MAP_SIZE];  // Array for storing keys (e.g., hashed edge identifiers)
//     Edge values[HASH_MAP_SIZE]; // Array for storing edge values
//     unsigned short active_edge_count; // Tracks the number of active edges
//     EdgeBlock *lptr;  // Pointer to the left block
//     EdgeBlock *rptr;  // Pointer to the right block
//     EdgeBlock *level_order_predecessor;

//     __device__ int hashFunction(int key) const {
//         // Simple modulo hash function
//         return key & (HASH_MAP_SIZE - 1); // Efficient when HASH_MAP_SIZE is a power of 2
//     }

//     __device__ bool insert(int key, const Edge &value) {
//         int index = hashFunction(key);
//         for (int i = 0; i < HASH_MAP_SIZE; ++i) {
//             int probeIndex = (index + i) & (HASH_MAP_SIZE - 1); // Linear probing
//             if (keys[probeIndex] == -1 || keys[probeIndex] == key) {
//                 keys[probeIndex] = key;
//                 values[probeIndex] = value;
//                 ++active_edge_count;
//                 return true;
//             }
//         }
//         return false; // Hashmap is full
//     }

//     __device__ Edge* lookup(int key) {
//         int index = hashFunction(key);
//         for (int i = 0; i < HASH_MAP_SIZE; ++i) {
//             int probeIndex = (index + i) & (HASH_MAP_SIZE - 1); // Linear probing
//             if (keys[probeIndex] == -1) {
//                 return nullptr; // Key not found
//             }
//             if (keys[probeIndex] == key) {
//                 return &values[probeIndex]; // Key found
//             }
//         }
//         return nullptr; // Key not found
//     }

//     __device__ bool remove(int key) {
//         int index = hashFunction(key);
//         for (int i = 0; i < HASH_MAP_SIZE; ++i) {
//             int probeIndex = (index + i) & (HASH_MAP_SIZE - 1); // Linear probing
//             if (keys[probeIndex] == -1) {
//                 return false; // Key not found
//             }
//             if (keys[probeIndex] == key) {
//                 keys[probeIndex] = -1; // Mark as deleted
//                 --active_edge_count;
//                 return true;
//             }
//         }
//         return false; // Key not found
//     }

//     // Initialize the hashmap
//     __device__ void initialize() {
//         for (int i = 0; i < HASH_MAP_SIZE; ++i) {
//             keys[i] = -1; // Use -1 to represent an empty slot
//         }
//         active_edge_count = 0;
//     }
// };

*/

struct EdgePreallocatedQueue{
    unsigned long long front;
    unsigned long long rear;
    unsigned long count;
    EdgeBlock **edge_block_address;
};

struct VertexDictionary {
    unsigned long *vertex_id;
    unsigned long *edge_block_count;
    unsigned long active_vertex_count;

    unsigned int *active_edge_count;
    unsigned long *last_insert_edge_offset;

    EdgeBlock **last_insert_edge_block;
    EdgeBlock **edge_block_address;
};

#endif