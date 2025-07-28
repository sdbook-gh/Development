cmake_minimum_required(VERSION 4.0)
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 添加 DLL 目标
add_library(ImageLibrary SHARED 
    src/ImageLibrary.cpp
    bridge/ImageLibrary_wrap.cxx
)
# 设置头文件目录
target_include_directories(ImageLibrary PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)
