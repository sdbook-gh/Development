set(CMAKE_CXX_STANDARD 17)
add_executable(test_shmallocator test_shmallocator.cpp shmallocator.cpp)
target_compile_options(test_shmallocator PRIVATE -g -Wall)
target_link_libraries(test_shmallocator PRIVATE pthread)
