cmake_minimum_required(VERSION 3.29)
project(untitled1 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(untitled1 atividade02Parelelo.cu)

set_target_properties(untitled1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
