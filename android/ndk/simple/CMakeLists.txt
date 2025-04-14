cmake_minimum_required(VERSION 3.22)

add_library(
    test_ndk
    SHARED
    test_ndk.cpp
)
find_library(
    log-lib
    log
)
target_link_libraries(
    test_ndk
    ${log-lib}
)

