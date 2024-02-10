set(CMAKE_CXX_STANDARD 17)

set(ASAN_OPTIONS -fsanitize=address -fsanitize=leak -fsanitize=undefined -fsanitize-address-use-after-scope -fno-sanitize=null) # -static-libasan
# set(ASAN_OPTIONS -fsanitize=thread -pthread) # -static-libasan

add_library(test_lib SHARED test_lib.cpp)
target_compile_options(test_lib PRIVATE -g -Wall ${ASAN_OPTIONS})
target_link_options(test_lib PRIVATE ${ASAN_OPTIONS})
target_link_libraries(test_lib PRIVATE pthread)

add_executable(test_asan test_asan.cpp)
target_compile_options(test_asan PRIVATE -g ${ASAN_OPTIONS})
target_link_options(test_asan PRIVATE ${ASAN_OPTIONS})
target_link_libraries(test_asan PRIVATE test_lib pthread)
