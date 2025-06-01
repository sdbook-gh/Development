cmake_minimum_required(VERSION 3.20)

project(CppDLLDemo VERSION 1.0)      # 项目名称和版本

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义宏以区分导出/导入（Windows 必需）
add_definitions(-DMATHLIBRARY_EXPORTS)

# 添加 DLL 目标
add_library(MathLibrary SHARED 
    src/MathLibrary.cpp
    include/MathLibrary.h
)

# 设置头文件目录
target_include_directories(MathLibrary PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 可选：设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
