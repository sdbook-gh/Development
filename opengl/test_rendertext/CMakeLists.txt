add_executable(test_rendertext test_rendertext.cpp glad.c)
target_include_directories(test_rendertext PRIVATE /usr/include/freetype2 include)
target_link_libraries(test_rendertext PRIVATE glfw GL freetype)
