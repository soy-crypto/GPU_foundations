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

    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(Tensor&& other) noexcept;

    float* get();

    size_t length() const;

    void fill(float value);

    float sum() const;
};