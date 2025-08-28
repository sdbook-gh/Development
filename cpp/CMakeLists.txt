cmake_minimum_required(VERSION ${CMAKE_VERSION})

set(CMAKE_CXX_STANDARD 17)
include_directories(.)
add_executable(test test.cpp)
add_executable(test_chrono test_chrono.cpp)
add_executable(test_macro test_macro.cpp)
add_executable(test_spmc_mpmc test_spmc_mpmc.cpp)

set(CMAKE_CXX_STANDARD 20)
add_executable(test_template test_template.cpp)
