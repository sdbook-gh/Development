#include "ImageLibrary.h"
#define STB_IMAGE_IMPLEMENTATION
#include <cstdarg>
#include <cstring>
#include <fstream>

#include "stb_image.h"

void log(const char* fmt, ...) {
  static char buffer[4096];
  static bool res = []() {
    std::ofstream clearFile("imagelibrary.log", std::ios::trunc);
    clearFile.close();
    return true;
  }();
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  // 追加写入日志文件
  static std::ofstream logFile("imagelibrary.log", std::ios::app);
  if (logFile) { logFile << buffer << std::endl; }
}

std::vector<uint8_t> ImageUtils::m_buffer;
ImageUtils::ImageUtils() { log("ImageUtils constructor called");getImageBufferPointer(); }
ImageUtils::~ImageUtils() { log("ImageUtils destructor called");getImageBufferPointer(); }
bool ImageUtils::loadImage() {
  if (!m_buffer.empty()) return true;
  std::vector<unsigned char> data;
  {
    std::ifstream file("e:/dev/SDK/unity_project/My project/image.jpg", std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize length = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(length);
    if (!file.read(reinterpret_cast<char*>(data.data()), length)) return false;
  }
  size_t length = data.size();
  int width, height, channels;
  unsigned char* pixels = stbi_load_from_memory((const stbi_uc*)&data[0], (int)length, &width, &height, &channels, 3);
  m_buffer.resize(width * height * 3);
  memcpy(&m_buffer[0], pixels, m_buffer.size());
  stbi_image_free(pixels);
  return true;
}
unsigned long ImageUtils::getImageBufferPointer() {
  if (!m_buffer.empty()) {
    log("Image buffer pointer: %lu", (unsigned long)&m_buffer[0]);
    return (unsigned long)&m_buffer[0];
  } else {
    log("Image buffer is empty, returning 0");
    return 0; // Return 0 if the buffer is empty
  }
}
int ImageUtils::getImageBufferSize() {
  if (!m_buffer.empty()) {
    log("Image buffer size: %d", (int)m_buffer.size());
    return (int)m_buffer.size();
  } else {
    log("Image buffer is empty, returning 0");
    return 0;
  }
}
