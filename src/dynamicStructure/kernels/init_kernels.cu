
#include "init_kernels.cuh"

__global__ void data_structure_init(VertexDictionary *device_vertex_dictionary, unsigned long *d_vertex_id, unsigned long *d_edge_block_count, unsigned int *d_active_edge_count, unsigned long *d_last_insert_edge_offset, EdgeBlock **d_last_insert_edge_block, EdgeBlock **d_edge_block_address, EdgeBlock **d_queue_edge_block_address) {
    d_e_queue.front = -1;
    d_e_queue.rear = -1;
    d_e_queue.count = 0;

    device_vertex_dictionary->active_vertex_count = 0;

    device_vertex_dictionary->vertex_id = d_vertex_id;
    device_vertex_dictionary->edge_block_count = d_edge_block_count;
    device_vertex_dictionary->active_edge_count = d_active_edge_count;
    device_vertex_dictionary->last_insert_edge_offset = d_last_insert_edge_offset;
    device_vertex_dictionary->last_insert_edge_block = d_last_insert_edge_block;
    device_vertex_dictionary->edge_block_address = d_edge_block_address;

    d_e_queue.edge_block_address = d_queue_edge_block_address;
    printf("D_Queue_Edge_block address : %p\n", d_queue_edge_block_address);
}

__global__ void parallel_push_edge_preallocate_list_to_device_queue(EdgeBlock *d_edge_preallocate_list, unsigned long total_edge_blocks_count_init, unsigned long device_edge_block_capacity) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < total_edge_blocks_count_init)
    {
        unsigned long free_blocks = device_edge_block_capacity - d_e_queue.count;

        // if ((free_blocks < total_edge_blocks_count_init) || (d_e_queue.rear + total_edge_blocks_count_init) % device_edge_block_capacity == d_e_queue.front) {
        //     return;
        // }

        d_e_queue.edge_block_address[id] = d_edge_preallocate_list + id;
        if(id < 10){
            printf("EdgeBlock-%ld address = [%p]\n", id, d_e_queue.edge_block_address[id]);
        }
    }
}

__global__ void parallel_push_queue_update(unsigned long total_edge_blocks_count_init) {
    if (d_e_queue.front == -1)
        d_e_queue.front = 0;

    d_e_queue.rear = (d_e_queue.rear + (unsigned int)total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE;
    d_e_queue.count += (unsigned int)total_edge_blocks_count_init;

    printf("Queue front is %lu(%p) and rear is %lu(%p)\n", d_e_queue.front, d_e_queue.edge_block_address[d_e_queue.front], d_e_queue.rear, d_e_queue.edge_block_address[d_e_queue.rear]);
}

__global__ void parallel_vertex_dictionary_init_v1(unsigned long vertex_size, VertexDictionary *device_vertex_dictionary) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < vertex_size) {
        if (id == 0)
            device_vertex_dictionary->active_vertex_count += vertex_size;
        device_vertex_dictionary->vertex_id[id] = id + 1;
    }

}

__global__ void printVertexDictionaryKernel(const VertexDictionary *d_vertex_dict, size_t vertex_size) {

   for(size_t idx = 0; idx < vertex_size; idx++) {
        // Print vertex_id for the specific vertex
        printf("Vertex[%lu]:\n", idx);
        printf("\tvertex_id: %lu\n", d_vertex_dict->vertex_id[idx]);
        printf("\tedge_block_count: %lu\n", d_vertex_dict->edge_block_count[idx]);
        printf("\tactive_edge_count: %u\n", d_vertex_dict->active_edge_count[idx]);
        printf("\tlast_insert_edge_offset: %lu\n", d_vertex_dict->last_insert_edge_offset[idx]);
        printf("\tlast_insert_edge_block address: %p\n", (void*)d_vertex_dict->last_insert_edge_block[idx]);
        printf("\tedge_block_address: %p\n", (void*)d_vertex_dict->edge_block_address[idx]);
        printf("\n");
    }
}

__global__ void printEdgeBlockQueue(EdgeBlock **d_queue_edge_block_address, size_t queue_size) {

    size_t sz = 40;
    if(queue_size < sz) sz = queue_size;

    for(int i = 0 ; i < sz; i++) {
        EdgeBlock *block = d_queue_edge_block_address[i];

        if (block != NULL) {
            printf("EdgeBlock[%lu] -> ", i);

            // Print edge block entries
            for (unsigned long j = 0; j < EDGE_BLOCK_SIZE; j++) {
                printf(" %lu ", block->edge_block_entry[j].destination_vertex);
            }
            printf("\n");
            // Add more fields to print if needed
        } else {
            printf("EdgeBlock[%lu]: NULL\n", i);
        }
    }
}
