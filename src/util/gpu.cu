#include "gpu.cuh"

int selectLeastLoadedGPU() {
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return -1;
    }

    int leastLoadedDevice = 0;
    size_t leastFreeMemory = SIZE_MAX;
    size_t freeMem = 0;
    size_t totalMem = 0;

    for (int i = 0; i < deviceCount; ++i) {
        CUDA_CALL(cudaSetDevice(i));

        cudaDeviceProp deviceProp;
        CUDA_CALL(cudaGetDeviceProperties(&deviceProp, i));

        
        CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));

        if (freeMem < leastFreeMemory) {
            leastFreeMemory = freeMem;
            leastLoadedDevice = i;
        }
    }
    std::cout << "\n[+] Available device Id is : " << leastLoadedDevice << " with available memory : " << leastFreeMemory / (1024 * 1024) << " MB\n";
    return leastLoadedDevice;
}

void printDeviceStatistics(int gpuId){

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, gpuId));

    // Print basic information
    std::cout << "Device Name: " << deviceProp.name << std::endl;
    std::cout << "CUDA Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

    // Print memory information
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Total Constant Memory: " << deviceProp.totalConstMem / 1024.0 << " KB" << std::endl;

    // Print other important information
    std::cout << "Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dimension (x, y, z): (" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Size (x, y, z): (" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;

    // Print clock rates
    std::cout << "Clock Rate: " << deviceProp.clockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory Clock Rate: " << deviceProp.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;

    // Print cache information
    std::cout << "L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;

    // Print additional memory alignment information
    std::cout << "Texture Alignment: " << deviceProp.textureAlignment << " bytes" << std::endl;
    std::cout << "Surface Alignment: " << deviceProp.texturePitchAlignment << " bytes" << std::endl;

    // Check for Unified Addressing (used for sharing memory between CPU and GPU)
    std::cout << "Unified Addressing: " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;

    // Print maximum number of concurrent kernels
    std::cout << "Max Concurrent Kernels: " << deviceProp.concurrentKernels << std::endl;

    // Print memory information at runtime
    size_t freeMem = 0, totalMem = 0;
    CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));

    std::cout << "Free Memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
}