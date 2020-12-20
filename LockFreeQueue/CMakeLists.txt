# cmake_minimum_required(VERSION <specify CMake version here>)
project(TestCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_DIR)
add_executable(TestCPP ${SRC_DIR})
target_link_libraries(TestCPP Threads::Threads -latomic)
