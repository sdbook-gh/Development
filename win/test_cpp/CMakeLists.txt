cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(test_cpp test_cpp.cpp)
target_include_directories(test_cpp PRIVATE include)  # 添加头文件目录
target_link_directories(test_cpp PRIVATE dll) # 添加 DLL 目录
target_link_libraries(test_cpp PRIVATE MathLibrary.dll)  # 链接 DLL
