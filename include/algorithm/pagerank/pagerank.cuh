#ifndef PAGERANK_CUH
#define PAGERANK_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"
#include "pr_kernel.cuh"

void static_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter, thrust::device_vector<float> &d_pageRankVector_1, thrust::device_vector<float> &d_pageRankVector_2);

void dynamic_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter, thrust::device_vector<float> &d_pageRankVector_1, thrust::device_vector<float> &d_pageRankVector_2);

#endif