cmake_minimum_required(VERSION 3.29)
project(Hillis_Steele CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Hillis_Steele main.cu)

set_target_properties(Hillis_Steele PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
