project(test)
set(CMAKE_C_FLAGS "-g")
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_LIST_DIR})
file(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/test_osg.cpp)
add_executable(test ${SRCS})
target_link_libraries(test osgGA osgDB osgViewer osg pthread)
