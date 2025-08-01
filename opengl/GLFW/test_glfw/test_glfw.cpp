// #include <GLFW/glfw3.h>
// #include <iostream>

// void error_callback(int error, const char *description) {
//   std::cerr << "Error " << error << ": " << description << std::endl;
// }

// int main() {
//   // 初始化GLFW
//   if (!glfwInit()) {
//     std::cerr << "Failed to initialize GLFW" << std::endl;
//     return -1;
//   }

//   // 设置错误回调
//   glfwSetErrorCallback(error_callback);

//   // 设置OpenGL版本为3.3
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

//   // 创建窗口
//   GLFWwindow *window =
//       glfwCreateWindow(800, 600, "GLFW Window", nullptr, nullptr);
//   if (!window) {
//     std::cerr << "Failed to create GLFW window" << std::endl;
//     glfwTerminate();
//     return -1;
//   }

//   // 设置当前上下文
//   glfwMakeContextCurrent(window);

//   // 渲染循环
//   while (!glfwWindowShouldClose(window)) {
//     // 清除颜色缓冲
//     glClear(GL_COLOR_BUFFER_BIT);
//     glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

//     // 交换缓冲并处理事件
//     glfwSwapBuffers(window);
//     glfwPollEvents();
//   }

//   // 清理资源
//   glfwDestroyWindow(window);
//   glfwTerminate();
//   return 0;
// }

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

// 顶点着色器源码
const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    gl_Position = vec4(aPos, 1.0);
    vColor = aColor;
}
)glsl";

// 片段着色器源码
const char* fragmentShaderSrc = R"glsl(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)glsl";

// 检查编译或链接错误
void checkCompileErrors(GLuint shader, const std::string& type) {
  GLint success;
  char infoLog[1024];
  if (type != "PROGRAM") {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
      std::cerr << "ERROR::SHADER_COMPILATION of type: " << type << "\n" << infoLog << "\n";
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
      std::cerr << "ERROR::PROGRAM_LINKING\n" << infoLog << "\n";
    }
  }
}

int main() {
  // 初始化 GLFW
  if (!glfwInit()) {
    std::cerr << "GLFW 初始化失败\n";
    return -1;
  }

  // 使用 OpenGL 3.3 核心模式
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // 创建窗口
  GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Trangle", nullptr, nullptr);
  if (!window) {
    std::cerr << "创建 GLFW 窗口失败\n";
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  // 初始化 GLEW
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "GLEW 初始化失败\n";
    glfwDestroyWindow(window);
    glfwTerminate();
    return -1;
  }

  // 顶点数据：位置 + 颜色
  float vertices[] = {
    // 位置           // 颜色
    0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, // 顶部（红）
    -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // 左下（绿）
    1.0f,  -1.0f, 0.0f, 0.0f, 0.0f, 1.0f // 右下（蓝）
  };

  // 创建 VAO、VBO
  GLuint VAO, VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  // 绑定并复制顶点数据
  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // 位置属性
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // 颜色属性
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // 编译着色器并创建程序
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSrc, nullptr);
  glCompileShader(vertexShader);
  checkCompileErrors(vertexShader, "VERTEX");

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSrc, nullptr);
  glCompileShader(fragmentShader);
  checkCompileErrors(fragmentShader, "FRAGMENT");

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  checkCompileErrors(shaderProgram, "PROGRAM");

  // 删除中间着色器对象
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // 渲染循环
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    // 使用着色器程序并绘制三角形
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // 清理资源
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteProgram(shaderProgram);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
