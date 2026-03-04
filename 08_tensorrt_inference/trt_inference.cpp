#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity,const char* msg) noexcept override
    {
        if(severity<=Severity::kINFO)
            std::cout<<msg<<std::endl;
    }
};

int main()
{
    Logger logger;

    auto builder =
        createInferBuilder(logger);

    auto network =
        builder->createNetworkV2(0);

    auto parser =
        nvonnxparser::createParser(*network,logger);

    parser->parseFromFile(
        "resnet18.onnx",
        (int)ILogger::Severity::kWARNING);

    auto config =
        builder->createBuilderConfig();

    auto engine =
        builder->buildEngineWithConfig(
            *network,
            *config
        );

    auto context =
        engine->createExecutionContext();

    std::cout<<"TensorRT engine built\n";
}