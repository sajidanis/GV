#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>

#include <stdio.h>
#include <cuda.h>

#define INF 1000000
#define WARP_SIZE 32

#define INF 1e9

// Kernel to initialize arrays
__global__ void initializeArrays(int n, int *sigma, float *delta, int *dist) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        sigma[tid] = 0;      // Number of shortest paths
        delta[tid] = 0.0f;   // Dependency values
        dist[tid] = INF;     // Distance array initialized to infinity
    }
}

// Kernel to perform BFS and compute shortest paths
__global__ void bfsKernel(int n, int *row_ptr, int *col_ind, int source, 
                          int *sigma, int *dist, int *queue, int *queue_size) {
    __shared__ int local_queue[1024];
    __shared__ int local_queue_size;

    if (threadIdx.x == 0) local_queue_size = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == source) {
        dist[tid] = 0;
        sigma[tid] = 1;
        queue[atomicAdd(queue_size, 1)] = tid;
    }

    __syncthreads();

    while (*queue_size > 0) {
        if (threadIdx.x == 0) local_queue_size = 0;
        __syncthreads();

        for (int i = threadIdx.x; i < *queue_size; i += blockDim.x) {
            int u = queue[i];
            for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
                int v = col_ind[j];
                if (atomicCAS(&dist[v], INF, dist[u] + 1) == INF) {
                    local_queue[atomicAdd(&local_queue_size, 1)] = v;
                }
                if (dist[v] == dist[u] + 1) {
                    atomicAdd(&sigma[v], sigma[u]);
                }
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            for (int i = 0; i < local_queue_size; i++) {
                queue[i] = local_queue[i];
            }
            *queue_size = local_queue_size;
        }
        __syncthreads();
    }
}


// Kernel function for BFS using CSR with warp-level optimizations
__global__ void bfsKernel(int n, int *row_ptr, int *col_ind, int source, 
                          int *sigma, int *dist, int *queue, int *queue_next, int *queue_size, int *queue_next_size) {

    __shared__ int local_queue[1024];  // Shared memory queue
    __shared__ int local_queue_size;   // Shared memory queue size

    if (threadIdx.x == 0) local_queue_size = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize BFS
    if (tid == source) {
        dist[tid] = 0;
        sigma[tid] = 1;
        queue[atomicAdd(queue_size, 1)] = tid;
    }

    __syncthreads();

    while (*queue_size > 0) {
        if (threadIdx.x == 0) local_queue_size = 0;
        __syncthreads();

        // Load-balancing: Each warp processes one node at a time
        int node_idx = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;

        for (int i = node_idx; i < *queue_size; i += blockDim.x / WARP_SIZE) {
            int u = queue[i];

            int start = row_ptr[u];
            int end = row_ptr[u + 1];

            // Each warp processes neighbors in parallel
            for (int j = start + lane; j < end; j += WARP_SIZE) {
                int v = col_ind[j];

                if (atomicCAS(&dist[v], INF, dist[u] + 1) == INF) {  // First time discovering v
                    int pos = atomicAdd(&local_queue_size, 1);
                    local_queue[pos] = v;  // Store v in local queue
                }

                if (dist[v] == dist[u] + 1) {  // Count shortest paths
                    atomicAdd(&sigma[v], sigma[u]);
                }
            }
        }

        __syncthreads();

        // Copy local queue to global queue
        if (threadIdx.x == 0) {
            for (int i = 0; i < local_queue_size; i++) {
                queue_next[i] = local_queue[i];
            }
            *queue_next_size = local_queue_size;
        }

        __syncthreads();

        // Swap queues
        if (threadIdx.x == 0) {
            int *temp = queue;
            queue = queue_next;
            queue_next = temp;
            *queue_size = *queue_next_size;
        }
        __syncthreads();
    }
}


// Kernel to compute dependencies and update betweenness centrality
__global__ void dependencyKernel(int n, int *row_ptr, int *col_ind, float *delta,
                                 int *sigma, int *dist, float *bc) {
    __shared__ float shared_delta[1024];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n && dist[tid] != INF) {
        shared_delta[threadIdx.x] = delta[tid];
        for (int j = row_ptr[tid]; j < row_ptr[tid + 1]; j++) {
            int v = col_ind[j];
            if (dist[v] == dist[tid] + 1) {
                float contrib = ((float)sigma[tid] / sigma[v]) * (1.0f + delta[v]);
                atomicAdd(&shared_delta[threadIdx.x], contrib);
            }
        }
        delta[tid] += shared_delta[threadIdx.x];
        if (tid != blockIdx.x)
            atomicAdd(&bc[tid], delta[tid]);
    }
}

// Host function to compute betweenness centrality using CUDA
void betweennessCentralityCUDA(int n, const std::vector<int> &row_ptr,
                               const std::vector<int> &col_ind,
                               std::vector<float> &bc) {
    // Device memory allocation
    int *d_row_ptr, *d_col_ind;
    cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int));
    cudaMalloc(&d_col_ind, col_ind.size() * sizeof(int));
    
    cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind.data(), col_ind.size() * sizeof(int), cudaMemcpyHostToDevice);

    float *d_bc;
    cudaMalloc(&d_bc, n * sizeof(float));
    cudaMemset(d_bc, 0, n * sizeof(float));

    // Temporary arrays for each source vertex
    int *d_sigma, *d_dist;
    float *d_delta;
    cudaMalloc(&d_sigma, n * sizeof(int));
    cudaMalloc(&d_dist, n * sizeof(int));
    cudaMalloc(&d_delta, n * sizeof(float));

    // Queue for BFS traversal
    int *d_queue;
    cudaMalloc(&d_queue, n * sizeof(int));
    
    int h_queue_size = 0;
    int *d_queue_size;
    cudaMalloc(&d_queue_size, sizeof(int));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Iterate over all source vertices
    for (int source = 0; source < n; ++source) {
        // Initialize arrays
        initializeArrays<<<grid, block>>>(n, d_sigma, d_delta, d_dist);

        // Reset queue size
        cudaMemcpy(d_queue_size, &h_queue_size, sizeof(int), cudaMemcpyHostToDevice);

        // Perform BFS from the source vertex
        bfsKernel<<<grid, block>>>(n, d_row_ptr, d_col_ind,
                                   source, d_sigma,
                                   d_dist, d_queue,
                                   d_queue_size);

        // Compute dependencies and update BC scores
        dependencyKernel<<<grid, block>>>(n, d_row_ptr,
                                          d_col_ind,
                                          d_delta,
                                          d_sigma,
                                          d_dist,
                                          d_bc);
    }

    // Copy results back to host
    cudaMemcpy(bc.data(), d_bc, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_bc);
    cudaFree(d_sigma);
    cudaFree(d_dist);
    cudaFree(d_delta);
}
