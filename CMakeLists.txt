project(cudatest1)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

option(CUDA_VERBOSE_PTXAS
    "Enable verbose output from the PTXAS assembler"
    OFF)

cuda_add_executable(cudatest1 main.c)
