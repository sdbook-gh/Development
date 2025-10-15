cmake_minimum_required(VERSION ${CMAKE_VERSION})
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -UANDROID")

add_executable(test_rga src/test_rga.cpp)
target_include_directories(test_rga PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_directories(test_rga PRIVATE ${CMAKE_CURRENT_LIST_DIR}/lib)
target_link_libraries(test_rga rga)
