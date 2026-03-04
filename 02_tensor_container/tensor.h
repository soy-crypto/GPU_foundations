#pragma once

#include <cstddef>

class Tensor {

private:

    float* data;
    size_t size;

public:

    Tensor(size_t n);

    ~Tensor();

    Tensor(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);

    Tensor& operator=(Tensor&& other) noexcept;

    void fill(float value);

    float sum() const;

};