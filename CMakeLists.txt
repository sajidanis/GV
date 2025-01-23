cmake_minimum_required(VERSION 3.28)

project(GraphVine CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/algorithm/pagerank)
include_directories(${CMAKE_SOURCE_DIR}/include/dynamicStructure)
include_directories(${CMAKE_SOURCE_DIR}/include/dynamicStructure/kernels)
include_directories(${CMAKE_SOURCE_DIR}/include/reader)
include_directories((${CMAKE_SOURCE_DIR}/include/util))

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(GraphVine src/main.cu
    src/reader/marketReader.cu
    src/util/profiler.cu
    src/dynamicStructure/graphvine.cu
    src/dynamicStructure/csr.cu
    src/util/util.cu
    src/dynamicStructure/kernels/init_kernels.cu
    src/dynamicStructure/kernels/insert_kernels.cu
    src/util/device_functions.cu
    src/algorithm/pagerank/pagerank.cu
    src/algorithm/pagerank/pr_kernel.cu
)

set_target_properties(GraphVine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)