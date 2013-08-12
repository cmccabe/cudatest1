project(cudatest1)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

option(CUDA_VERBOSE_PTXAS
    "Enable verbose output from the PTXAS assembler"
    OFF)

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_HOST_COMPILATION_CPP OFF)

set(CMAKE_BUILD_TYPE, Release) # can also be Debug
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fno-strict-aliasing -rdynamic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_REENTRANT -D_FILE_OFFSET_BITS=64")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GNU_SOURCE")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wuninitialized")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0 -fstack-protector")
set(CMAKE_CURRENT_BINARY_DIR, build)
SET(CMAKE_INSTALL_PREFIX /usr/local)

cuda_add_executable(cudatest1 main.cu)
