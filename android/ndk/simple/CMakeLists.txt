cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

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

add_executable(test_asan test_asan.cpp)
target_compile_options(test_asan PRIVATE -g -fsanitize=hwaddress -fsanitize=leak -fno-omit-frame-pointer)
target_link_options(test_asan PRIVATE -fsanitize=hwaddress -fsanitize=leak -fno-omit-frame-pointer)
target_link_libraries(test_asan PRIVATE log)
