cmake_minimum_required(VERSION 3.29)
project(untitled3 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(untitled3 main.cu)

set_target_properties(untitled3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
