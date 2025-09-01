cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_CXX_STANDARD 17)
add_executable(test_DMIPS test_DMIPS.cpp)
target_compile_options(test_DMIPS PRIVATE -g -O3 -fno-inline)

add_executable(test_func_duration test_func_duration.cpp)
target_compile_options(test_func_duration PRIVATE -g -O3 -fno-inline -mavx2)

add_executable(convert_perf_timestamp convert_perf_timestamp.cpp)
