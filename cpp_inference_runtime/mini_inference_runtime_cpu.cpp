#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

////////////////////////////////////////////////////////////
// Tensor
////////////////////////////////////////////////////////////
class Tensor
{
    private:
        std::vector<float> data;
        int rows;
        int cols;

    public:
        Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

        float& operator()(int r, int c)
        {
            return data[r * cols + c];
        }

        float operator()(int r, int c) const
        {
            return data[r * cols + c];
        }

        float* getData() { return data.data(); }
        const float* getData() const { return data.data(); }

        int getRows() const { return rows; }
        int getCols() const { return cols; }
        int getSize() const { return rows * cols; }

};

////////////////////////////////////////////////////////////
// Operator Interface
////////////////////////////////////////////////////////////

class Operator
{
    public:
        virtual ~Operator() {}
        virtual Tensor forward(const Tensor& input) = 0;
};

////////////////////////////////////////////////////////////
// ReLU
////////////////////////////////////////////////////////////

class ReLU : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            const float* in = input.getData();
            float* out = output.getData();

            for (int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::max(0.0f, in[i]);
            }

            return output;
        }

};

////////////////////////////////////////////////////////////
// Softmax
////////////////////////////////////////////////////////////

class Softmax : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            Tensor output(input.getRows(), input.getCols());

            const float* in = input.getData();
            float* out = output.getData();

            float maxVal = in[0];
            for (int i = 1; i < input.getSize(); i++)
            {
                maxVal = std::max(maxVal, in[i]);
            }

            float sum = 0.0f;
            for (int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::exp(in[i] - maxVal);
                sum += out[i];
            }

            for (int i = 0; i < input.getSize(); i++)
            {
                out[i] /= sum;
            }

            return output;
        }

};

////////////////////////////////////////////////////////////
// Graph
////////////////////////////////////////////////////////////

class Graph
{
    private:
        std::vector<Operator*> ops;

    public:
        void add_op(Operator* op)
        {
            ops.push_back(op);
        }

        Tensor run(const Tensor& input)
        {
            Tensor x = input;

            for (const auto* op : ops)
            {
                Tensor out = op->forward(x);
                x = std::move(out);   // 🔥 move instead of copy
            }

            return x;
        }
        
};

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////

int main()
{
    Tensor input(1, 3);
    float* data = input.getData();

    for (int i = 0; i < input.getSize(); i++)
    {
        data[i] = static_cast<float>(i);
    }

    Graph graph;
    ReLU relu;
    Softmax softmax;

    graph.add_op(&relu);
    graph.add_op(&softmax);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    double latency =
        std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Output: ";
    float* out = output.getData();
    for (int i = 0; i < output.getSize(); i++)
    {
        std::cout << out[i] << " ";
    }

    std::cout << "\nLatency: " << latency << " ms\n";
    std::cout << "Inference done\n";

    return 0;
}