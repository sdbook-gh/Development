set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SRCS src/*.cpp)
add_executable(app ${SRCS})
target_include_directories(app PRIVATE open/include include)
target_link_directories(app PRIVATE open/lib)
target_link_libraries(app unwind dep_lib)
