cmake_minimum_required(VERSION ${CMAKE_VERSION})

set(CMAKE_CXX_STANDARD 17)
include_directories(.)
add_executable(test test.cpp)
add_executable(test_chrono test_chrono.cpp)
add_executable(test_macro test_macro.cpp)
add_executable(test_spmc_mpmc test_spmc_mpmc.cpp)

set(CMAKE_CXX_STANDARD 20)
add_executable(test_template test_template.cpp)

add_executable(test_virtual_func test_virtual_func.cpp)
target_compile_options(test_virtual_func PRIVATE -g -fno-inline -rdynamic)
target_link_libraries(test_virtual_func PRIVATE -ldl)

add_executable(test_futex test_futex.cpp)

add_executable(test_time_wheel test_time_wheel.cpp)
