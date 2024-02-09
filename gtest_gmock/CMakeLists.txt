set(CMAKE_CXX_STANDARD 17)
add_executable(test_cpp test.cpp)
target_include_directories(test_cpp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/gtest/include)
target_compile_options(test_cpp PRIVATE -g)
target_link_directories(test_cpp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/gtest/lib)
target_link_libraries(test_cpp PRIVATE gtest gmock pthread)

