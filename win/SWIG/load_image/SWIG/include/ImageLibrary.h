#pragma once

#include <cstdint>
#include <vector>

class ImageUtils {
public:
  ImageUtils();
  ~ImageUtils();
  bool loadImage();
  unsigned long getImageBufferPointer();
  int getImageBufferSize();
private:
  static std::vector<uint8_t> m_buffer;
};
