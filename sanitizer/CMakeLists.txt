set(CMAKE_CXX_STANDARD 17)
add_link_options(-fuse-ld=gold) # speed up link

set(ASAN_OPTIONS -fsanitize=address -fsanitize=leak -fsanitize=undefined -fsanitize-address-use-after-scope -fno-sanitize=null -fno-omit-frame-pointer) # -static-libasan
# set(ASAN_OPTIONS -fsanitize=thread -fno-omit-frame-pointer) # -static-libasan

add_library(test_sanitizer_lib SHARED test_sanitizer_lib.cpp)
target_compile_options(test_sanitizer_lib PRIVATE -g -Wall ${ASAN_OPTIONS})
target_link_options(test_sanitizer_lib PRIVATE ${ASAN_OPTIONS})
target_link_libraries(test_sanitizer_lib PRIVATE pthread)

add_executable(test_sanitizer test_sanitizer.cpp)
target_compile_options(test_sanitizer PRIVATE -g ${ASAN_OPTIONS})
target_link_options(test_sanitizer PRIVATE ${ASAN_OPTIONS})
target_link_libraries(test_sanitizer PRIVATE test_sanitizer_lib pthread)
