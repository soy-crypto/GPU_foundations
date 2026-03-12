#include "tensor.h"

//constructor
Tensor::Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

//operator
float& Tensor::operator()(int r, int c) 
{
    return data[r * cols + c];
}

//const operator
float Tensor::operator()(int r, int c) const 
{
    return data[r * cols + c];
}