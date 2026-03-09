#include "kernels.h"

__global__ void coalesced_read(float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        output[idx] = input[idx];
    }
        
}