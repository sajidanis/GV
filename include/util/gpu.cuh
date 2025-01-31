#ifndef GPU_CUH
#define GPU_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#include "cudaError.cuh"

int selectLeastLoadedGPU();

void printDeviceStatistics(int gpuId);

#endif