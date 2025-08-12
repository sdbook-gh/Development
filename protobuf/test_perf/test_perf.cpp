#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "iguana/pb_reader.hpp"
#include "iguana/pb_writer.hpp"
#include "image.pb.h"
#include "pointcloud.pb.h"
#include "pointcloud_generated.h"
#include "sdproto.h"

struct SPBImageData {
  std::string data;
};
YLT_REFL(SPBImageData, data)

void ser_img_to_vec(std::vector<uint8_t> &out_vec, uint8_t key, const std::vector<uint8_t> &img_data) {
  auto enc_size = [](uint64_t size, std::vector<uint8_t> &buffer) {
    uint32_t count{0};
    while (size > 0x7F) {
      buffer[count++] = static_cast<uint8_t>((size & 0x7F) | 0x80);
      size >>= 7;
    }
    buffer[count++] = static_cast<uint8_t>(size);
    return count;
  };
  static std::vector<uint8_t> variant_vec;
  variant_vec.resize(10);
  uint32_t size = enc_size(img_data.size(), variant_vec);
  out_vec.resize(1 + size + img_data.size());
  out_vec[0] = key;
  memcpy(&out_vec[1], &variant_vec[0], size);
  memcpy(&out_vec[1 + size], &img_data[0], img_data.size());
}

int old_main() {
  constexpr int TEST_COUNT = 1000;
  std::vector<uint8_t> buf;
  buf.resize(1920 * 1080 * 3); // 假设图像为 1920x1080 RGB
  // std::string spb_buf((char *)&buf[0], (char *)&spb_buf[buf.size()-1]);
  std::for_each(buf.begin(), buf.end(), [](uint8_t &byte) {
    byte = rand() % 256; // 填充随机数据模拟图像
  });

  std::vector<uint8_t> out_buf;
  std::string spb_out_buf;
  {
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_COUNT; ++i) {
      out_buf.resize(buf.size());
      memcpy(out_buf.data(), buf.data(), buf.size());
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "memcpy time for TEST_COUNT iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average memcpy time: " << serialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
  }
  {
    ImageData msg;
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_COUNT; ++i) {
      msg.set_data(buf.data(), buf.size());
      out_buf.resize(msg.ByteSizeLong());
      google::protobuf::io::ArrayOutputStream aos(out_buf.data(), out_buf.size());
      if (!msg.SerializeToZeroCopyStream(&aos)) {
        std::cerr << "Serialize failed\n";
        return -1;
      }
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "Prorobuf serialization time for TEST_COUNT iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average protobuf serialization time: " << serialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
  }
  {
    SPBImageData msg;
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_COUNT; ++i) {
      msg.data = std::string((char *)&buf[0], (char *)&buf[buf.size() - 1]);
      spb_out_buf.clear();
      iguana::to_pb(msg, spb_out_buf);
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "SPB serialization time for TEST_COUNT iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average SPB serialization time: " << serialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
  }
  {
    auto serialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_COUNT; ++i) {
      ser_img_to_vec(out_buf, (1 << 3) | 2, buf);
      // static bool _tmp = [&] {
      //   printf("out_buf size: %lu\n", out_buf.size());
      //   return true;
      // }();
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "Manual prorobuf serialization time for TEST_COUNT iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average manual protobuf serialization time: " << serialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
  }

  {
    ImageData msg;
    auto deserialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_COUNT; ++i) {
      google::protobuf::io::ArrayInputStream ais(out_buf.data(), out_buf.size());
      if (!msg.ParseFromZeroCopyStream(&ais)) {
        std::cerr << "Parse failed\n";
        return -1;
      }
    }
    auto deserialize_end = std::chrono::high_resolution_clock::now();
    auto deserialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(deserialize_end - deserialize_start);
    std::cout << "Protobuf deserialization time: " << deserialize_duration.count() << " ms" << std::endl;
    std::cout << "Average protobuf dserialization time: " << deserialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
  }
  {
    SPBImageData msg;
    auto deserialize_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_COUNT; ++i) { iguana::from_pb(msg, spb_out_buf); }
    auto deserialize_end = std::chrono::high_resolution_clock::now();
    auto deserialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(deserialize_end - deserialize_start);
    std::cout << "SPB deserialization time: " << deserialize_duration.count() << " ms" << std::endl;
    std::cout << "Average SPB dserialization time: " << deserialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
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
    for (int i = 0; i < TEST_COUNT; ++i) {
      google::protobuf::io::ArrayOutputStream aos(out_buf.data(), out_buf.size());
      if (!msg.SerializeToZeroCopyStream(&aos)) {
        std::cerr << "Serialize failed\n";
        return -1;
      }
    }
    auto serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_end - serialize_start);
    std::cout << "Small data protobuf serialization time for TEST_COUNT iterations: " << serialize_duration.count() << " ms" << std::endl;
    std::cout << "Average small data protobuf serialization time: " << serialize_duration.count() / (double)TEST_COUNT << " ms" << std::endl;
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
    std::cout << "Small data protobuf deserialization time: " << deserialize_duration.count() << " ms" << std::endl;
    std::cout << "Average small data protobuf deserialization time: " << deserialize_duration.count() / 100.0 << " ms" << std::endl;
  }
  return 0;
}

using apollo::drivers::PointCloud;
using apollo::drivers::PointCloudOpt;
using apollo::drivers::PointCloudRefine;
using apollo::drivers::PointXYZITRefine;
using fbs::apollo::drivers::PointCloudT;
using fbs::apollo::drivers::PointXYZITT;
using namespace std::chrono;

#pragma push
#pragma pack(1)
struct PointCloudData {
  double x{0}, y{0}, z{0};
  double intensity{0};
  uint64_t timestamp{0};
};
#pragma pop

// 生成随机点云数据vector
std::vector<PointCloudData> GenerateRandomPointCloudDataVector(size_t num_points) {
  std::vector<PointCloudData> points(num_points);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> coord(-100.0, 100.0);
  std::uniform_real_distribution<double> intensity_dist(0.0, 255.0);
  std::uniform_int_distribution<uint64_t> ts(0, 1e9);
  for (size_t i = 0; i < num_points; ++i) {
    points[i].x = coord(rng);
    points[i].y = coord(rng);
    points[i].z = coord(rng);
    points[i].intensity = intensity_dist(rng);
    points[i].timestamp = ts(rng);
  }
  return points;
}

PointCloud MakePointCloudFromData(const std::vector<PointCloudData> &point_data) {
  PointCloud cloud;
  cloud.mutable_header()->set_timestamp_sec(1234567890.123);
  cloud.set_frame_id("velodyne64");
  cloud.set_is_dense(true);
  cloud.set_width(point_data.size());
  cloud.set_height(1);
  cloud.set_measurement_time(0.123);
  // 预分配空间
  cloud.mutable_point()->Reserve(point_data.size());
  // 遍历 PointCloudData 向量，将数据添加到 PointCloud 中
  for (const auto &data : point_data) {
    auto *point = cloud.add_point();
    point->set_x(static_cast<float>(data.x));
    point->set_y(static_cast<float>(data.y));
    point->set_z(static_cast<float>(data.z));
    point->set_intensity(static_cast<uint32_t>(data.intensity));
    point->set_timestamp(data.timestamp);
  }
  return cloud;
}

PointCloudOpt MakePointCloudOptFromData(const std::vector<PointCloudData> &point_data) {
  PointCloudOpt cloud;
  cloud.mutable_header()->set_timestamp_sec(1234567890.123);
  cloud.set_frame_id("velodyne64");
  cloud.set_is_dense(true);
  cloud.set_width(point_data.size());
  cloud.set_height(1);
  cloud.set_measurement_time(0.123);
  cloud.set_point((const char *)&point_data[0], point_data.size() * sizeof(PointCloudData));
  return cloud;
}

PointCloudRefine MakePointCloudRefineFromData(const std::vector<PointCloudData> &point_data) {
  PointCloudRefine cloud;
  cloud.mutable_header()->set_timestamp_sec(1234567890.123);
  cloud.set_frame_id("velodyne64");
  cloud.set_is_dense(true);
  cloud.set_width(point_data.size());
  cloud.set_height(1);
  cloud.set_measurement_time(0.123);
  // 预分配空间
  cloud.mutable_point()->Reserve(point_data.size());
  // 下面用通俗的语言告诉你：
  // “当 PointXYZITRefine 的所有 5 个字段都被赋了值以后，它到底在二进制里长什么样？如果我把这些字节一一拆开，能看到什么？”
  // 一、先记住 protobuf 的“小套路”
  // 1. 数据是一小块一小块（key-value）拼接起来的，不固定长度。
  // 2. 每个 key 都是一个 varint，低 3 位存“线型”（wire type），其余位存“字段号”。
  // 3. 出现顺序就是字段号顺序；字段号不连续也没关系。
  // 4. 默认值不占空间，但题目说“所有字段都有值”，所以 5 个字段都会出现。
  // 二、把字段号和线型先列出来
  // | 字段 | 类型 | 字段号 | 线型 | 二进制怎么存 |
  // | x | double | 1 | 64-bit (1) | little-endian 8 字节 |
  // | y | double | 2 | 64-bit (1) | little-endian 8 字节 |
  // | z | double | 3 | 64-bit (1) | little-endian 8 字节 |
  // | intensity | fixed32 | 4 | 32-bit (5) | little-endian 4 字节 |
  // | timestamp | fixed64 | 5 | 64-bit (1) | little-endian 8 字节 |
  // 三、算一下 key（field tag）
  // 公式：tag = (field_number << 3) | wire_type
  // - x： (1<<3)|1 = 0x09
  // - y： (2<<3)|1 = 0x11
  // - z： (3<<3)|1 = 0x19
  // - intensity： (4<<3)|5 = 0x25
  // - timestamp： (5<<3)|1 = 0x29
  // 四、给一个具体例子
  // 假设我们给 5 个字段随意赋了值：
  // x = 1.0
  // y = 2.0
  // z = 3.0
  // intensity = 0x12345678
  // timestamp = 0x123456789ABCDEF0
  // 它们的 IEEE-754 小端十六进制分别是
  // 1.0 → 00 00 00 00 00 00 F0 3F
  // 2.0 → 00 00 00 00 00 00 00 40
  // 3.0 → 00 00 00 00 00 00 08 40
  // intensity → 78 56 34 12
  // timestamp → F0 DE BC 9A 78 56 34 12
  // 五、把整条消息拼起来（十六进制）
  // 09 00 00 00 00 00 00 F0 3F
  // 11 00 00 00 00 00 00 00 40
  // 19 00 00 00 00 00 00 08 40
  // 25 78 56 34 12
  // 29 F0 DE BC 9A 78 56 34 12
  // 六、如果你拿放大镜看
  // 第 1 字节：0x09 → 字段 1，double → 后面 8 字节是 x
  // 第 10 字节：0x11 → 字段 2，double → 后面 8 字节是 y
  // 第 19 字节：0x19 → 字段 3，double → 后面 8 字节是 z
  // 第 28 字节：0x25 → 字段 4，fixed32 → 后面 4 字节是 intensity
  // 第 33 字节：0x29 → 字段 5，fixed64 → 后面 8 字节是 timestamp
  // 七、一句话总结
  // 二进制就是一串“字段号 + 类型”的 key，后面紧跟着该字段的小端原始字节；整条消息没有分隔符，也没有额外元数据，所以非常紧凑。
  // for (const auto &p : point_data) {
  //   auto *dst = ::new (cloud.mutable_point()->Add()) PointXYZITRefine;
  //   std::memcpy(dst, &p, sizeof(PointCloudData));
  // }
  for (const auto &data : point_data) {
    auto *point = cloud.add_point();
    point->set_x((data.x));
    point->set_y((data.y));
    point->set_z((data.z));
    point->set_intensity((data.intensity));
    point->set_timestamp(data.timestamp);
  }
  return cloud;
}

class PreallocatedAllocator : public flatbuffers::Allocator {
public:
  char *buffer_{nullptr};
  size_t size_{0};
  uint8_t *allocate(size_t size) override {
    if (size > size_) {
      printf("PreallocatedAllocator::allocate size %ld > size_ %ld\n", size, size_);
      return nullptr;
    }
    return (uint8_t *)buffer_;
  }
  void deallocate(uint8_t *p, size_t size) override {}
};

PointCloudT MakeFBSPointCloudFromData(const std::vector<PointCloudData> &point_data) {
  using namespace fbs::apollo::drivers;
  using namespace fbs::apollo::common;
  auto fbb = std::make_shared<flatbuffers::FlatBufferBuilder>();
  PointCloudT cloud;
  cloud.header = std::make_unique<HeaderT>();
  cloud.header->timestamp_sec = 1234567890.123;
  cloud.frame_id = "velodyne64";
  cloud.is_dense = true;
  cloud.width = point_data.size();
  cloud.height = 1;
  cloud.measurement_time = 0.123;
  cloud.point.resize(point_data.size());
  for (int i = 0; i < point_data.size(); ++i) {
    cloud.point[i] = std::make_unique<PointXYZITT>();
    cloud.point[i]->x = point_data[i].x;
    cloud.point[i]->y = point_data[i].y;
    cloud.point[i]->z = point_data[i].z;
    cloud.point[i]->intensity = point_data[i].intensity;
    cloud.point[i]->timestamp = point_data[i].timestamp;
  }
  return cloud;
}

sdproto::PointCloud MakeSDPointCloudFromData(const std::vector<PointCloudData> &point_data) {
  sdproto::PointCloud cloud;
  cloud.set_frame_id("velodyne64");
  cloud.set_is_dense(true);
  cloud.set_measurement_time(0.123);
  cloud.set_width(point_data.size());
  cloud.set_height(1);
  for (int i = 0; i < point_data.size(); ++i) {
    sdproto::PointXYZIT *point = cloud.add_point();
    point->set_x(point_data[i].x);
    point->set_y(point_data[i].y);
    point->set_z(point_data[i].z);
    point->set_intensity(point_data[i].intensity);
    point->set_timestamp(point_data[i].timestamp);
  }
  return cloud;
}

double BenchSerialize(const std::vector<PointCloudData> &point_data, std::vector<char> &buffer, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    PointCloud cloud = MakePointCloudFromData(point_data);
    buffer.resize(cloud.ByteSizeLong());
    cloud.SerializeToArray(buffer.data(), buffer.size());
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchOptSerialize(const std::vector<PointCloudData> &point_data, std::vector<char> &buffer, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    PointCloudOpt cloud = MakePointCloudOptFromData(point_data);
    buffer.resize(cloud.ByteSizeLong());
    cloud.SerializeToArray(buffer.data(), buffer.size());
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchRefineSerialize(const std::vector<PointCloudData> &point_data, std::vector<char> &buffer, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    PointCloudRefine cloud = MakePointCloudRefineFromData(point_data);
    buffer.resize(cloud.ByteSizeLong());
    cloud.SerializeToArray(buffer.data(), buffer.size());
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchFBSSerialize(const std::vector<PointCloudData> &point_data, std::vector<char> &buffer, size_t &offset, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    // 方式一
    {
      // buffer.resize(buffer.capacity());
      // PreallocatedAllocator allocator;
      // allocator.buffer_ = &buffer[0];
      // allocator.size_ = buffer.size();
      // flatbuffers::FlatBufferBuilder fbb(buffer.capacity(), &allocator);
      // PointCloudT cloud = MakeFBSPointCloudFromData(point_data);
      // fbb.Finish(fbs::apollo::drivers::PointCloud::Pack(fbb, &cloud));
      // offset = buffer.size() - fbb.GetSize();
    }

    // 方式二
    {
      buffer.resize(buffer.capacity());
      PreallocatedAllocator allocator;
      allocator.buffer_ = &buffer[0];
      allocator.size_ = buffer.size();
      flatbuffers::FlatBufferBuilder fbb(buffer.capacity(), &allocator);
      auto header = fbs::apollo::common::CreateHeader(fbb, 1234567890.123);
      std::vector<flatbuffers::Offset<fbs::apollo::drivers::PointXYZIT>> fbs_points;
      fbs_points.reserve(point_data.size());
      for (const auto &p : point_data) { fbs_points.push_back(fbs::apollo::drivers::CreatePointXYZIT(fbb, p.x, p.y, p.z, p.intensity, p.timestamp)); }
      auto vec = fbb.CreateVector(fbs_points);
      auto pc = fbs::apollo::drivers::CreatePointCloud(fbb, header, fbb.CreateString("velodyne64"), true, vec, 0.123, point_data.size(), 1);
      fbb.Finish(pc);
      offset = buffer.size() - fbb.GetSize();
      // memmove(buffer.data(), fbb.GetBufferPointer(), fbb.GetSize());
      // buffer.resize(fbb.GetSize());
    }
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchSDSerialize(const std::vector<PointCloudData> &point_data, std::vector<char> &buffer, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    sdproto::PointCloud cloud = MakeSDPointCloudFromData(point_data);
    cloud.serializeTo(buffer);
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchDeserialize(const std::vector<char> &buffer, PointCloud &out, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    out.Clear();
    out.ParseFromArray(buffer.data(), buffer.size());
    // 计算 intensity 最大的点
    uint32_t max_intensity = 0;
    for (const auto &point : out.point()) {
      if (point.intensity() > max_intensity) { max_intensity = point.intensity(); }
    }
    static bool res = [max_intensity, i] {
      std::cout << "Max intensity: " << max_intensity << std::endl;
      return true;
    }();
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchOptDeserialize(const std::vector<char> &buffer, PointCloudOpt &out, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    out.Clear();
    out.ParseFromArray(buffer.data(), buffer.size());
    // 计算 intensity 最大的点
    uint32_t max_intensity = 0;
    const PointCloudData *point_array = reinterpret_cast<const PointCloudData *>(out.point().data());
    size_t point_count = out.point().size() / sizeof(PointCloudData);
    for (size_t i = 0; i < point_count; ++i) {
      if (static_cast<uint32_t>(point_array[i].intensity) > max_intensity) { max_intensity = static_cast<uint32_t>(point_array[i].intensity); }
    }
    static bool res = [max_intensity, i] {
      std::cout << "Max intensity: " << max_intensity << std::endl;
      return true;
    }();
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchRefineDeserialize(const std::vector<char> &buffer, PointCloudRefine &out, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    out.Clear();
    out.ParseFromArray(buffer.data(), buffer.size());
    // 计算 intensity 最大的点
    uint32_t max_intensity = 0;
    const PointXYZITRefine *point_array = reinterpret_cast<const PointXYZITRefine *>(out.mutable_point()->data());
    size_t point_count = out.mutable_point()->size();
    for (size_t i = 0; i < point_count; ++i) {
      if (point_array[i].intensity() > max_intensity) { max_intensity = point_array[i].intensity(); }
    }
    std::cout << "Max intensity: " << max_intensity << std::endl;
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchFBSDeserialize(const std::vector<char> &buffer, size_t offset, PointCloudT &out, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto cloud = flatbuffers::GetRoot<fbs::apollo::drivers::PointCloud>(&buffer[offset]);
    cloud->UnPackTo(&out);
    // 计算 intensity 最大的点
    uint32_t max_intensity = 0;
    for (const auto &point : out.point) {
      if (point->intensity > max_intensity) { max_intensity = point->intensity; }
    }
    static bool res = [max_intensity, i] {
      std::cout << "Max intensity: " << max_intensity << std::endl;
      return true;
    }();
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

double BenchSDDeserialize(std::vector<char> &buffer, sdproto::PointCloud &out, int iterations = 100) {
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    out.deserializeFrom(buffer);

    // 计算 intensity 最大的点
    uint32_t max_intensity = 0;
    for (const auto &point : out.points()) {
      if (point.intensity() > max_intensity) { max_intensity = point.intensity(); }
    }
    static bool res = [max_intensity, i] {
      std::cout << "Max intensity: " << max_intensity << std::endl;
      return true;
    }();
  }
  auto t1 = high_resolution_clock::now();
  double sec = duration_cast<duration<double>>(t1 - t0).count();
  return sec;
}

int main(int argc, char *argv[]) {
  size_t num_points = 100000;
  int iterations = 200;
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  std::cout << "Generating cloud with " << num_points << " points...\n";
  auto point_data = GenerateRandomPointCloudDataVector(num_points);
  std::vector<char> buffer;
  buffer.reserve(10 * 1024 * 1024);
  {
    std::cout << "\n=== Serialization ===\n";
    double ser_sec = BenchSerialize(point_data, buffer, iterations);
    std::cout << "  " << iterations << " ops in " << ser_sec * 1000 << " ms\n";
  }
  {
    std::cout << "\n=== Deserialization ===\n";
    PointCloud dummy;
    double deser_sec = BenchDeserialize(buffer, dummy, iterations);
    std::cout << "  " << iterations << " ops in " << deser_sec * 1000 << " ms\n";
  }
  {
    std::cout << "\n=== Opt Serialization ===\n";
    double ser_sec = BenchOptSerialize(point_data, buffer, iterations);
    std::cout << "  " << iterations << " ops in " << ser_sec * 1000 << " ms\n";
  }
  {
    std::cout << "\n=== Opt Deserialization ===\n";
    PointCloudOpt dummy;
    double deser_sec = BenchOptDeserialize(buffer, dummy, iterations);
    std::cout << "  " << iterations << " ops in " << deser_sec * 1000 << " ms\n";
  }
  // {
  //   std::cout << "\n=== Refine Serialization ===\n";
  //   double ser_sec = BenchRefineSerialize(point_data, buffer, iterations);
  //   std::cout << "  " << iterations << " ops in " << ser_sec * 1000 << " ms\n";
  // }
  // {
  //   std::cout << "\n=== Refine Deserialization ===\n";
  //   PointCloudRefine dummy;
  //   double deser_sec = BenchRefineDeserialize(buffer, dummy, iterations);
  //   std::cout << "  " << iterations << " ops in " << deser_sec * 1000 << " ms\n";
  // }
  size_t offset = 0;
  {
    std::cout << "\n=== FBS Serialization ===\n";
    double ser_sec = BenchFBSSerialize(point_data, buffer, offset, iterations);
    std::cout << "  " << iterations << " ops in " << ser_sec * 1000 << " ms\n";
  }
  {
    std::cout << "\n=== FBS Deserialization ===\n";
    PointCloudT dummy;
    double deser_sec = BenchFBSDeserialize(buffer, offset, dummy, iterations);
    std::cout << "  " << iterations << " ops in " << deser_sec * 1000 << " ms\n";
  }
  {
    std::cout << "\n=== SD Serialization ===\n";
    double ser_sec = BenchSDSerialize(point_data, buffer, iterations);
    std::cout << "  " << iterations << " ops in " << ser_sec * 1000 << " ms\n";
  }
  {
    std::cout << "\n=== SD Deserialization ===\n";
    sdproto::PointCloud dummy;
    double deser_sec = BenchSDDeserialize(buffer, dummy, iterations);
    std::cout << "  " << iterations << " ops in " << deser_sec * 1000 << " ms\n";
  }
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
