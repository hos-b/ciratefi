cmake_minimum_required(VERSION 3.1)
project(ciratefi)

set(CMAKE_CXX_FLAGS "-O0 -g -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 

include_directories(
	include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/main.cpp
                               src/filters/circ_sample_filter.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)