#include <iostream>

class Buffer {

private:
    float* data;
    int size;

public:

    Buffer(int n) : size(n)
    {
        data = new float[n];
        std::cout<<"Allocated\n";
    }

    ~Buffer()
    {
        delete[] data;
        std::cout<<"Freed\n";
    }

    void fill(float v)
    {
        for(int i=0;i<size;i++)
            data[i] = v;
    }

    float sum()
    {
        float s = 0;

        for(int i=0;i<size;i++)
            s += data[i];

        return s;
    }
};

int main()
{
    Buffer A(10);

    A.fill(1);

    std::cout<<"Sum "<<A.sum()<<"\n";
}