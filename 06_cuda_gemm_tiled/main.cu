#include <iostream>
#include <cuda_runtime.h>

#define TILE 16

__global__
void gemm_tiled(float* A,float* B,float* C,int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0;

    for(int t=0;t<N/TILE;t++)
    {
        As[threadIdx.y][threadIdx.x] =
            A[row*N + t*TILE + threadIdx.x];

        Bs[threadIdx.y][threadIdx.x] =
            B[(t*TILE + threadIdx.y)*N + col];

        __syncthreads();

        for(int k=0;k<TILE;k++)
            sum += As[threadIdx.y][k] *
                   Bs[k][threadIdx.x];

        __syncthreads();
    }

    C[row*N+col] = sum;
}

int main()
{
    int N = 1024;
    size_t bytes = N*N*sizeof(float);

    float *A,*B,*C;

    cudaMallocManaged(&A,bytes);
    cudaMallocManaged(&B,bytes);
    cudaMallocManaged(&C,bytes);

    for(int i=0;i<N*N;i++)
    {
        A[i]=1;
        B[i]=1;
    }

    dim3 threads(TILE,TILE);
    dim3 blocks(N/TILE,N/TILE);

    gemm_tiled<<<blocks,threads>>>(A,B,C,N);

    cudaDeviceSynchronize();

    std::cout<<"C[0] "<<C[0]<<std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}