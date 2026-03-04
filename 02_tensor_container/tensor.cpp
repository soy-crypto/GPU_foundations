#include "tensor.h"
#include <algorithm>

Tensor::Tensor(size_t n) : size(n)
{
    data = new float[n];
}

Tensor::~Tensor()
{
    delete[] data;
}

Tensor::Tensor(const Tensor& other)
{
    size = other.size;

    data = new float[size];

    std::copy(other.data,other.data+size,data);
}

Tensor::Tensor(Tensor&& other) noexcept
{
    data = other.data;

    size = other.size;

    other.data = nullptr;

    other.size = 0;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if(this!=&other)
    {
        delete[] data;

        size = other.size;

        data = new float[size];

        std::copy(other.data,other.data+size,data);
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if(this!=&other)
    {
        delete[] data;

        data = other.data;

        size = other.size;

        other.data = nullptr;

        other.size = 0;
    }

    return *this;
}

void Tensor::fill(float v)
{
    for(size_t i=0;i<size;i++)
        data[i]=v;
}

float Tensor::sum() const
{
    float s=0;

    for(size_t i=0;i<size;i++)
        s+=data[i];

    return s;
}