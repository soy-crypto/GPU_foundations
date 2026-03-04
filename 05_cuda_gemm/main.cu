__global__
void matmul(float* A,float* B,float* C,int N)
{
    int row=
        blockIdx.y*blockDim.y
        +threadIdx.y;

    int col=
        blockIdx.x*blockDim.x
        +threadIdx.x;

    if(row<N && col<N)
    {
        float sum=0;

        for(int k=0;k<N;k++)
            sum+=A[row*N+k]*B[k*N+col];

        C[row*N+col]=sum;
    }
}