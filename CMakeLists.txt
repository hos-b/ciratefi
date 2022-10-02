cmake_minimum_required(VERSION 3.1)
project(ciratefi)

list(APPEND CMAKE_PREFIX_PATH $ENV{HOME}/lib/libtorch/share/cmake)

set(CMAKE_CXX_FLAGS "-O0 -g -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)

include_directories(
	include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/main.cpp
                               src/filters/circ_sample_filter.cpp
                               src/filters/radi_sample_filter.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}
                                      ${TORCH_LIBRARIES}
                                      fmt::fmt
                                      OpenMP::OpenMP_CXX)