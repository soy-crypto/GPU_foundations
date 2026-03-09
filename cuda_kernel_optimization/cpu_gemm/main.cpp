#include <iostream>
#include <vector>
#include <chrono>

void gemm(const float* A, const float* B, float* C, int N)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum=0;
            for(int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[ k * N + j];
            }

            C[i * N + j]=sum;
        }

    }
        
}

int main()
{
    //Init
    int N = 512;
    std::vector<float> A(N * N, 1);
    std::vector<float> B(N * N, 1);
    std::vector<float> C(N * N, 0);

    //Gemm
    auto start = std::chrono::high_resolution_clock::now();
    gemm(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    //Update latency
    std::chrono::duration<double> diff=end-start;
    std::cout<<"Runtime "<<diff.count()<<" s\n";

    //Return
    return;
}