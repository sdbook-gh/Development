#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "iguana/pb_reader.hpp"
#include "iguana/pb_writer.hpp"
#include "image.pb.h"

struct SPBImageData {
  std::string data;
};
YLT_REFL(SPBImageData, data)

int main() {
  // {
  //   SPBImageData p{"test"};
  //   std::string pb;
  //   iguana::to_pb(p, pb);
  //   SPBImageData p1;
  //   iguana::from_pb(p1, pb);
  //   printf("%s\n", p1.data.c_str());
  //   ImageData msg;
  //   google::protobuf::io::ArrayInputStream ais(pb.data(), pb.size());
  //   if (!msg.ParseFromZeroCopyStream(&ais)) {
  //     std::cerr << "Parse failed\n";
  //     return -1;
  //   }
  //   printf("%s\n", msg.data().c_str());
  // }
  std::vector<uint8_t> buf;
  buf.resize(1920 * 1080 * 3); // 假设图像为 1920x1080 RGB
  // std::string spb_buf((char *)&buf[0], (char *)&spb_buf[buf.size()-1]);
  std::for_each(buf.begin(), buf.end(), [](uint8_t &byte) {
    byte = rand() % 256; // 填充随机数据模拟图像
  });

  std::vector<uint8_t> out_buf;
  std::string spb_out_buf;
  {
    out_buf.resize(buf.size());
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) { memcpy(out_buf.data(), buf.data(), buf.size()); }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "memcpy time for 100 iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average memcpy time: " << serialize_duration.count() / 100.0 << " ms" << std::endl;
  }
  {
    ImageData msg;
    msg.set_data(buf.data(), buf.size());
    out_buf.resize(msg.ByteSizeLong());
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      google::protobuf::io::ArrayOutputStream aos(out_buf.data(), out_buf.size());
      if (!msg.SerializeToZeroCopyStream(&aos)) {
        std::cerr << "Serialize failed\n";
        return -1;
      }
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "Serialization time for 100 iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average serialization time: " << serialize_duration.count() / 100.0 << " ms" << std::endl;
  }
  {
    SPBImageData msg;
    msg.data = std::string((char *)&buf[0], (char *)&buf[buf.size()-1]);
    spb_out_buf.clear();
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1; ++i) {
      iguana::to_pb(msg, spb_out_buf);
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "SPB Serialization time for 100 iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average SPB serialization time: " << serialize_duration.count() / 100.0 << " ms" << std::endl;
  }
  {
    ImageData msg;
    auto deserialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      google::protobuf::io::ArrayInputStream ais(out_buf.data(), out_buf.size());
      if (!msg.ParseFromZeroCopyStream(&ais)) {
        std::cerr << "Parse failed\n";
        return -1;
      }
    }
    auto deserialize_end = std::chrono::high_resolution_clock::now();
    auto deserialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(deserialize_end - deserialize_start);
    std::cout << "Deserialization time: " << deserialize_duration.count() << " ms" << std::endl;
  }
  {
    SPBImageData msg;
    auto deserialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      iguana::from_pb(msg, spb_out_buf);
    }
    auto deserialize_end = std::chrono::high_resolution_clock::now();
    auto deserialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(deserialize_end - deserialize_start);
    std::cout << "SPB Deserialization time: " << deserialize_duration.count() << " ms" << std::endl;
  }

  buf.resize(640 * 480 * 3); // 假设图像为 1920x1080 RGB
  std::for_each(buf.begin(), buf.end(), [](uint8_t &byte) {
    byte = rand() % 256; // 填充随机数据模拟图像
  });
  {
    ImageData msg;
    msg.set_data(buf.data(), buf.size());
    out_buf.resize(msg.ByteSizeLong());
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      google::protobuf::io::ArrayOutputStream aos(out_buf.data(), out_buf.size());
      if (!msg.SerializeToZeroCopyStream(&aos)) {
        std::cerr << "Serialize failed\n";
        return -1;
      }
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "Serialization time for 100 iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average serialization time: " << serialize_duration.count() / 100.0 << " ms" << std::endl;
  }
  {
    ImageData msg;
    auto deserialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      google::protobuf::io::ArrayInputStream ais(out_buf.data(), out_buf.size());
      if (!msg.ParseFromZeroCopyStream(&ais)) {
        std::cerr << "Parse failed\n";
        return -1;
      }
    }
    auto deserialize_end = std::chrono::high_resolution_clock::now();
    auto deserialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(deserialize_end - deserialize_start);
    std::cout << "Deserialization time: " << deserialize_duration.count() << " ms" << std::endl;
  }
  return 0;
}
