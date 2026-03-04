#include "tensor.h"
#include <iostream>
#include <chrono>

int main() 
{
    //Base Check

    //Init
    size_t N = 10000000;
    Tensor A(N);
    A.fill(1.0f);
    auto start = std::chrono::high_resolution_clock::now();
    float result = A.sum();
    auto end = std::chrono::high_resolution_clock::now();

    //Print
    std::chrono::duration<double> diff = end - start;
    std::cout << "Sum: " << result << std::endl;
    std::cout << "Time: " << diff.count() << " seconds\n";

    //Return
    return 0;
    
}