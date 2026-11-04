cmake_minimum_required(VERSION 4.3)
project(untitled)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

add_executable(untitled main.cpp)
target_link_libraries(untitled PRIVATE OpenMP::OpenMP_CXX)
target_link_options(untitled PRIVATE ${OpenMP_CXX_FLAGS})
