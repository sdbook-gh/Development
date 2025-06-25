#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>

int decode_jpeg_file(const char* filename) {
  FILE* file = NULL;
  unsigned char* jpegBuf = NULL;
  unsigned char* imgBuf = NULL;
  tjhandle tjInstance = NULL;
  int width, height, jpegSubsamp, jpegColorspace;
  unsigned long jpegSize;
  int pixelFormat = TJPF_RGB;
  int retval = 0;

  // 打开 JPEG 文件
  file = fopen(filename, "rb");
  if (!file) {
    printf("Error: Cannot open file %s\n", filename);
    return -1;
  }

  // 获取文件大小
  fseek(file, 0, SEEK_END);
  jpegSize = ftell(file);
  fseek(file, 0, SEEK_SET);

  // 分配内存并读取文件
  jpegBuf = (unsigned char*)malloc(jpegSize);
  if (!jpegBuf) {
    printf("Error: Memory allocation failed\n");
    fclose(file);
    return -1;
  }

  if (fread(jpegBuf, jpegSize, 1, file) < 1) {
    printf("Error: Cannot read file\n");
    free(jpegBuf);
    fclose(file);
    return -1;
  }
  fclose(file);

  // 初始化 TurboJPEG 解码器
  tjInstance = tjInitDecompress();
  if (!tjInstance) {
    printf("Error: TurboJPEG init failed\n");
    free(jpegBuf);
    return -1;
  }

  // 读取 JPEG 头信息
  tjDecompressHeader3(tjInstance, jpegBuf, jpegSize, &width, &height, &jpegSubsamp, &jpegColorspace);
  printf("Image Info: %dx%d, Subsampling: %d, Colorspace: %d\n", width, height, jpegSubsamp, jpegColorspace);

  // 分配解码后的图像缓冲区
  imgBuf = (unsigned char*)malloc(width * height * tjPixelSize[pixelFormat]);
  if (!imgBuf) {
    printf("Error: Image buffer allocation failed\n");
    tjDestroy(tjInstance);
    free(jpegBuf);
    return -1;
  }

  // 解码 JPEG 图像
  if (tjDecompress2(tjInstance, jpegBuf, jpegSize, imgBuf, width, 0, height, pixelFormat, TJFLAG_FASTDCT) < 0) {
    printf("Error: JPEG decompression failed: %s\n", tjGetErrorStr());
    retval = -1;
  } else {
    printf("JPEG decompression successful!\n");
    printf("Decoded image size: %d bytes\n", width * height * tjPixelSize[pixelFormat]);

    // 这里可以使用解码后的图像数据 (imgBuf)
    // 例如保存为其他格式或进行图像处理
  }

  // 清理资源
  tjDestroy(tjInstance);
  free(jpegBuf);
  free(imgBuf);

  return retval;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <jpeg_file>\n", argv[0]);
    return 1;
  }

  return decode_jpeg_file(argv[1]);
}
