// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <GL/glew.h>
// #include <GLFW/glfw3.h>
// #include <jpeglib.h>

// // 错误回调函数
// void error_callback(int error, const char* description) {
//     std::cerr << "Error: " << description << std::endl;
// }

// // 读取JPEG文件函数
// bool load_jpeg(const char* filename, std::vector<unsigned char>& image,
//                unsigned int& width, unsigned int& height) {

//     struct jpeg_decompress_struct cinfo;
//     struct jpeg_error_mgr jerr;

//     FILE* file = fopen(filename, "rb");
//     if (!file) {
//         std::cerr << "Error opening JPEG file: " << filename << std::endl;
//         return false;
//     }

//     cinfo.err = jpeg_std_error(&jerr);
//     jpeg_create_decompress(&cinfo);
//     jpeg_stdio_src(&cinfo, file);
//     jpeg_read_header(&cinfo, TRUE);
//     jpeg_start_decompress(&cinfo);

//     width = cinfo.output_width;
//     height = cinfo.output_height;
//     unsigned int channels = cinfo.output_components;

//     image.resize(width * height * channels);
//     unsigned char* rowptr = image.data();

//     while (cinfo.output_scanline < cinfo.output_height) {
//         jpeg_read_scanlines(&cinfo, &rowptr, 1);
//         rowptr += width * channels;
//     }

//     jpeg_finish_decompress(&cinfo);
//     jpeg_destroy_decompress(&cinfo);
//     fclose(file);
//     return true;
// }

// // 创建纹理函数
// GLuint create_texture(const std::vector<unsigned char>& image,
//                       unsigned int width, unsigned int height) {

//     GLuint textureID;
//     glGenTextures(1, &textureID);
//     glBindTexture(GL_TEXTURE_2D, textureID);

//     // 设置纹理参数
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//     // 加载图像数据到纹理
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
//                  GL_RGB, GL_UNSIGNED_BYTE, image.data());

//     glGenerateMipmap(GL_TEXTURE_2D);
//     return textureID;
// }

// int main() {
//     // 初始化GLFW
//     if (!glfwInit()) {
//         std::cerr << "Failed to initialize GLFW" << std::endl;
//         return -1;
//     }

//     glfwSetErrorCallback(error_callback);

//     // 创建窗口
//     GLFWwindow* window = glfwCreateWindow(800, 600, "JPEG Texture Rendering", NULL, NULL);
//     if (!window) {
//         std::cerr << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }

//     glfwMakeContextCurrent(window);

//     // 初始化GLEW
//     if (glewInit() != GLEW_OK) {
//         std::cerr << "Failed to initialize GLEW" << std::endl;
//         return -1;
//     }

//     // 加载JPEG图像
//     std::vector<unsigned char> image_data;
//     unsigned int width, height;
//     const char* jpeg_file = "image.jpg"; // 替换为你的JPEG文件路径

//     if (!load_jpeg(jpeg_file, image_data, width, height)) {
//         std::cerr << "Failed to load JPEG image" << std::endl;
//         return -1;
//     }

//     // 创建纹理
//     GLuint texture = create_texture(image_data, width, height);

//     // 顶点数据 (位置, 纹理坐标)
//     float vertices[] = {
//         // 位置          // 纹理坐标
//         -0.5f, -0.5f,    0.0f, 0.0f,
//          0.5f, -0.5f,    1.0f, 0.0f,
//          0.5f,  0.5f,    1.0f, 1.0f,
//         -0.5f,  0.5f,    0.0f, 1.0f
//     };

//     unsigned int indices[] = {
//         0, 1, 2,
//         2, 3, 0
//     };

//     // 创建VAO, VBO, EBO
//     GLuint VAO, VBO, EBO;
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glGenBuffers(1, &EBO);

//     glBindVertexArray(VAO);

//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

//     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
//     glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

//     // 位置属性
//     glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);

//     // 纹理坐标属性
//     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
//     glEnableVertexAttribArray(1);

//     // 渲染循环
//     while (!glfwWindowShouldClose(window)) {
//         glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
//         glClear(GL_COLOR_BUFFER_BIT);

//         // 绑定纹理
//         glBindTexture(GL_TEXTURE_2D, texture);

//         // 绘制四边形
//         glBindVertexArray(VAO);
//         glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

//         glfwSwapBuffers(window);
//         glfwPollEvents();
//     }

//     // 清理资源
//     glDeleteVertexArrays(1, &VAO);
//     glDeleteBuffers(1, &VBO);
//     glDeleteBuffers(1, &EBO);
//     glDeleteTextures(1, &texture);

//     glfwTerminate();
//     return 0;
// }

#include <cstdio>
#include <cstring>
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <jpeglib.h>

// JPEG 加载函数：读取 JPEG 文件到内存，返回像素数据和宽高
bool loadJPEG(const char* filename, unsigned char*& outData, int& width, int& height) {
  FILE* infile = std::fopen(filename, "rb");
  if (!infile) {
    std::cerr << "无法打开 JPEG 文件: " << filename << std::endl;
    return false;
  }

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  width = cinfo.output_width;
  height = cinfo.output_height;
  int channels = cinfo.output_components; // 通常为 3 (RGB)

  size_t rowStride = width * channels;
  outData = new unsigned char[width * height * channels];

  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, rowStride, 1);

  unsigned char* ptr = outData;
  while (cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);
    std::memcpy(ptr, buffer[0], rowStride);
    ptr += rowStride;
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  std::fclose(infile);

  return true;
}

int main(int argc, char** argv) {
  unsigned char* imageData = nullptr;
  int imgW = 0, imgH = 0;
  if (!loadJPEG("image.jpg", imageData, imgW, imgH)) { return -1; }

  // 初始化 GLFW
  if (!glfwInit()) {
    std::cerr << "GLFW 初始化失败" << std::endl;
    delete[] imageData;
    return -1;
  }

  // 创建窗口与 OpenGL 上下文
  GLFWwindow* window = glfwCreateWindow(imgW, imgH, "OpenGL Jpeg", nullptr, nullptr);
  if (!window) {
    std::cerr << "窗口创建失败" << std::endl;
    glfwTerminate();
    delete[] imageData;
    return -1;
  }
  glfwMakeContextCurrent(window);

  // 初始化 GLEW
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "GLEW 初始化失败" << std::endl;
    glfwDestroyWindow(window);
    glfwTerminate();
    delete[] imageData;
    return -1;
  }

  // 创建纹理
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imgW, imgH, 0, GL_RGB, GL_UNSIGNED_BYTE, imageData);

  // 纹理参数
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  delete[] imageData; // 数据已上传到 GPU

  // 渲染循环
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    // 在正交投影下渲染全屏四边形
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, imgW, 0, imgH, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);

    glBegin(GL_QUADS);
    // glTexCoord2f(0.0f, 0.0f);
    // glVertex2f(0.0f, 0.0f);
    // glTexCoord2f(1.0f, 0.0f);
    // glVertex2f(imgW, 0.0f);
    // glTexCoord2f(1.0f, 1.0f);
    // glVertex2f(imgW, imgH);
    // glTexCoord2f(0.0f, 1.0f);
    // glVertex2f(0.0f, imgH);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(imgW, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(imgW, imgH);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, imgH);
    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteTextures(1, &tex);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
