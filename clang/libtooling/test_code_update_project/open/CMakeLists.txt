set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)

file(GLOB_RECURSE SRCS src/*.cpp)
add_library(dep_lib STATIC ${SRCS})
target_include_directories(dep_lib PRIVATE include)
target_link_options(dep_lib PRIVATE -fPIC -rdynamic)
