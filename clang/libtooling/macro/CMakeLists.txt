set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
link_directories(${CLANG_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(analyzer analyzer.cpp)
target_link_libraries(analyzer clangAST clangBasic clangFrontend clangIndex clangRewrite clangTooling ${LLVM_LIBRARIES} ${CLANG_LIBRARIES})

add_executable(matcher matcher.cpp)
target_link_libraries(matcher clangAST clangBasic clangFrontend clangIndex clangRewrite clangTooling ${LLVM_LIBRARIES} ${CLANG_LIBRARIES})
