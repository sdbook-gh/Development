find_package(glog REQUIRED)
add_library(test_lib STATIC test_lib.cpp)
target_compile_options(test_lib PRIVATE -g)
target_link_libraries(test_lib PRIVATE glog::glog)
