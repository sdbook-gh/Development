add_executable(test_renderjpeg test_renderjpeg.cpp)
target_link_libraries(test_renderjpeg PRIVATE glfw GL GLEW jpeg)
