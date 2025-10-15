
#include <cstdint>
#include <fstream>
#include <memory>
#include <vector>

#undef ANDROID
#include "RgaUtils.h"
#include "im2d.hpp"

int main() {
  IM_STATUS ret{IM_STATUS_NOERROR};
  int src_width{0}, src_height{0}, src_format{0};
  int dst_width{0}, dst_height{0}, dst_format{0};
  std::vector<uint8_t> src_buf;
  std::vector<uint8_t> dst_buf;
  rga_buffer_t src_img{0}, dst_img{0};
  rga_buffer_handle_t src_handle{0}, dst_handle{0};
  std::shared_ptr<int*> free_handle{nullptr, [&](auto) {
                                      if (src_handle) releasebuffer_handle(src_handle);
                                      if (dst_handle) releasebuffer_handle(dst_handle);
                                    }};
  src_width = 2688;
  src_height = 1520;
  src_format = RK_FORMAT_YCbCr_420_SP;
  dst_width = 1920;
  dst_height = 1080;
  dst_format = RK_FORMAT_RGB_888;
  src_buf.resize(src_width * src_height * get_bpp_from_format(src_format));
  dst_buf.resize(dst_width * dst_height * get_bpp_from_format(dst_format));
  /* fill image data */
  std::ifstream src_file("2k.rgb", std::ios::binary);
  if (!src_file.is_open()) {
    printf("src image open err\n");
    return -1;
  }
  src_file.read((char*)&src_buf[0], src_buf.size());
  src_file.close();
  src_handle = importbuffer_virtualaddr(&src_buf[0], src_buf.size());
  dst_handle = importbuffer_virtualaddr(&dst_buf[0], dst_buf.size());
  if (src_handle == 0 || dst_handle == 0) {
    printf("importbuffer failed!\n");
    return -1;
  }
  src_img = wrapbuffer_handle(src_handle, src_width, src_height, src_format);
  dst_img = wrapbuffer_handle(dst_handle, dst_width, dst_height, dst_format);
  ret = imcheck(src_img, dst_img, {}, {});
  if (IM_STATUS_NOERROR != ret) {
    printf("%d, check error! %s", __LINE__, imStrError(ret));
    return -1;
  }
  ret = imresize(src_img, dst_img);
  if (ret == IM_STATUS_SUCCESS) {
    printf("running success!\n");
  } else {
    printf("running failed, %s\n", imStrError((IM_STATUS)ret));
    return -1;
  }
  std::ofstream dst_file("1080.rgb", std::ios::binary);
  if (!dst_file.is_open()) {
    printf("dst image open err\n");
    return -1;
  }
  dst_file.write((char*)&dst_buf[0], dst_buf.size());
  dst_file.close();
  printf("dst image write success!\n");
  return ret;
}
