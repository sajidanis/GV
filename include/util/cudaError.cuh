#ifndef CUDAERROR_CUH
#define CUDAAERROR_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CALL(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                      \
    do {                                                                         \
        cudaError_t error = cudaGetLastError();                                  \
        if (error != cudaSuccess) {                                              \
            std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(error)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#endif