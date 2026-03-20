#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

class Tensor
{
    private:
        std::vector<float> data;
        int rows;
        int cols;

    public:
        Tensor(int r, int c): rows(r), cols(c), data(r * c, 0.0f) {}
        
        float& operator()(int r, int c) { return data[r * cols + c]; }
        float operator()(int r, int c) const { return data[r * cols + c]; }

        float* getData() { return data.data(); }
        const float* getData() const { return data.data(); }
        
        int getRows() const { return rows; }
        int getCols() const { return cols; }
        int getSize() const { return rows * cols; }

};


class Operator
{
    public:
        virtual ~Operator() {}
        virtual Tensor forward(const Tensor& input) = 0;
};


class ReLU : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            //Get in and out
            Tensor output(input.getRows(), input.getCols());
            const float* in = input.getData();
            float* out = output.getData();
            
            //ReLU
            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::max(0.0f, in[i]);
            }

            //Return
            return output;
        }

};


class Softmax : public Operator
{
    public:
        Tensor forward(const Tensor& input) override
        {
            //Init
            Tensor output(input.getRows(), input.getCols());
            float* out = output.getData();
            const float* in = input.getData();

            // Softmax
            float maxVal = in[0];
            for(int i = 0; i < input.getSize(); i++)
            {
                maxVal = std::max(maxVal, in[i]);
            }

            // Sum
            float sum = 0.0f;
            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] = std::exp(in[i] - maxVal);
                sum += out[i];
            }

            for(int i = 0; i < input.getSize(); i++)
            {
                out[i] /= sum;
            }


            //Return
            return output;
        }

};


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
            for(const auto* op : ops)
            {
                Tensor out = op->forward(x);
                x = std::move(out);
            }

            return x;

        }

};


int main()
{
    //Init
    /** input */
    Tensor input(1, 3);
    float* data = input.getData();
    for(int i = 0; i < input.getSize(); i++)
    {
        data[i] = static_cast<float>(i);
    }

    /*  Graph */
    Graph graph;
    ReLU relu;
    Softmax softmax;
    
    graph.add_op(&relu);
    graph.add_op(&softmax);

    //Forward
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = graph.run(input);
    auto end = std::chrono::high_resolution_clock::now();

    //Latency
    double latency = std::chrono::duration<double, std::milli>(end - start).count();

    //Print
    std::cout << "Output:" << std::endl;
    float* out = output.getData();
    for(int i = 0; i < output.getSize(); i++)
    {
        std::cout << out[i] << " ";
    }

    std::cout << "\n Latency " << latency << " ms\n";

    //Return
    return 0;

}