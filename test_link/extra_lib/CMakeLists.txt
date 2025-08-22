cmake_minimum_required(VERSION ${CMAKE_VERSION})
find_package(glog REQUIRED)
add_library(extra_lib STATIC extra_lib.cpp)
# -fPIC is used to link .a for .so
target_compile_options(extra_lib PRIVATE -g -fPIC)
target_link_libraries(extra_lib PRIVATE glog::glog)
