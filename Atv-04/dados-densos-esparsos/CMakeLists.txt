cmake_minimum_required(VERSION 3.29)
project(dados_densos_esparsos CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(dados_densos_esparsos main.cu)

set_target_properties(dados_densos_esparsos PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
