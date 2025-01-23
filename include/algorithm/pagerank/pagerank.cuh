#ifndef PAGERANK_CUH
#define PAGERANK_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphvine.cuh"
#include "pr_kernel.cuh"

void static_pagerank(GraphVine *graph, float alpha, float epsilon, int max_iter);

#endif