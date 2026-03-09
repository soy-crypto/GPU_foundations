cmake_minimum_required(VERSION 3.10)
project(cpp_inference_runtime)

set(CMAKE_CXX_STANDARD 17)

add_executable(inference_runtime main.cpp graph.cpp ops.cpp tensor.cpp)