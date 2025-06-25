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

add_executable(test_libjpegturbo test_libjpegturbo.cpp)
target_include_directories(test_libjpegturbo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo/_install/include)
target_link_directories(test_libjpegturbo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo/_install/lib)
target_link_libraries(test_libjpegturbo libturbojpeg.a)
