# cmake_minimum_required(VERSION <specify CMake version here>)
project(TestCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
add_executable(TestCPP main.cpp LockFreeQueue.h LockFreeLinkedQueue.h)
target_link_libraries(TestCPP Threads::Threads -latomic)
