#include <GLFW/glfw3.h>
#include <iostream>

void error_callback(int error, const char *description) {
  std::cerr << "Error " << error << ": " << description << std::endl;
}

int main() {
  // 初始化GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  // 设置错误回调
  glfwSetErrorCallback(error_callback);

  // 设置OpenGL版本为3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // 创建窗口
  GLFWwindow *window =
      glfwCreateWindow(800, 600, "GLFW Window", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  // 设置当前上下文
  glfwMakeContextCurrent(window);

  // 渲染循环
  while (!glfwWindowShouldClose(window)) {
    // 清除颜色缓冲
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // 交换缓冲并处理事件
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // 清理资源
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
