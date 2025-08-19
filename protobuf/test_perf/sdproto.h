#pragma once

// https://claude.ai/chat/ea4665e3-eb4a-4a01-a616-dae4ea2ec809
#include <stdint.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace sdproto {
class ProtobufSerializer {
private:
  // Protobuf wire types
  enum WireType { VARINT = 0, FIXED64 = 1, LENGTH_DELIMITED = 2, FIXED32 = 5 };
  // Helper functions for varint encoding
  static size_t varint_size(uint64_t value) {
    if (value < (1ULL << 7)) return 1;
    if (value < (1ULL << 14)) return 2;
    if (value < (1ULL << 21)) return 3;
    if (value < (1ULL << 28)) return 4;
    if (value < (1ULL << 35)) return 5;
    if (value < (1ULL << 42)) return 6;
    if (value < (1ULL << 49)) return 7;
    if (value < (1ULL << 56)) return 8;
    if (value < (1ULL << 63)) return 9;
    return 10;
  }
  static char* encode_varint(char* ptr, uint64_t value) {
    while (value >= 0x80) {
      *ptr++ = (char)(value | 0x80);
      value >>= 7;
    }
    *ptr++ = (char)value;
    return ptr;
  }
  static char* encode_fixed32(char* ptr, uint32_t value) {
    memcpy(ptr, &value, sizeof(uint32_t));
    return ptr + sizeof(uint32_t);
  }
  static char* encode_fixed64(char* ptr, uint64_t value) {
    memcpy(ptr, &value, sizeof(uint64_t));
    return ptr + sizeof(uint64_t);
  }
  static char* encode_float(char* ptr, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(float));
    return encode_fixed32(ptr, bits);
  }
  static char* encode_double(char* ptr, double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(double));
    return encode_fixed64(ptr, bits);
  }
  static char* encode_string(char* ptr, const std::string& str) {
    ptr = encode_varint(ptr, str.length());
    memcpy(ptr, str.data(), str.length());
    return ptr + str.length();
  }
  static uint32_t make_tag(uint32_t field_number, WireType wire_type) { return (field_number << 3) | wire_type; }
  static size_t string_size(const std::string& str) { return varint_size(str.length()) + str.length(); }
public:
  struct PointXYZIT {
    float x = std::numeric_limits<float>::quiet_NaN();
    float y = std::numeric_limits<float>::quiet_NaN();
    float z = std::numeric_limits<float>::quiet_NaN();
    uint32_t intensity = 0;
    uint64_t timestamp = 0;
    // Check if field has non-default value
    bool has_x() const { return !std::isnan(x); }
    bool has_y() const { return !std::isnan(y); }
    bool has_z() const { return !std::isnan(z); }
    bool has_intensity() const { return intensity != 0; }
    bool has_timestamp() const { return timestamp != 0; }
    size_t calculate_size() const {
      size_t size = 0;
      if (has_x()) { size += varint_size(make_tag(1, FIXED32)) + 4; }
      if (has_y()) { size += varint_size(make_tag(2, FIXED32)) + 4; }
      if (has_z()) { size += varint_size(make_tag(3, FIXED32)) + 4; }
      if (has_intensity()) { size += varint_size(make_tag(4, VARINT)) + varint_size(intensity); }
      if (has_timestamp()) { size += varint_size(make_tag(5, VARINT)) + varint_size(timestamp); }
      return size;
    }
    char* serialize(char* ptr) const {
      if (has_x()) {
        ptr = encode_varint(ptr, make_tag(1, FIXED32));
        ptr = encode_float(ptr, x);
      }
      if (has_y()) {
        ptr = encode_varint(ptr, make_tag(2, FIXED32));
        ptr = encode_float(ptr, y);
      }
      if (has_z()) {
        ptr = encode_varint(ptr, make_tag(3, FIXED32));
        ptr = encode_float(ptr, z);
      }
      if (has_intensity()) {
        ptr = encode_varint(ptr, make_tag(4, VARINT));
        ptr = encode_varint(ptr, intensity);
      }
      if (has_timestamp()) {
        ptr = encode_varint(ptr, make_tag(5, VARINT));
        ptr = encode_varint(ptr, timestamp);
      }
      return ptr;
    }
  };
  struct PointCloud {
    std::string frame_id;
    bool is_dense = false;
    std::vector<PointXYZIT> points;
    double measurement_time = 0.0;
    uint32_t width = 0;
    uint32_t height = 0;
    // Check if field has non-default value
    bool has_frame_id() const { return !frame_id.empty(); }
    bool has_is_dense() const { return is_dense != false; }
    bool has_points() const { return !points.empty(); }
    bool has_measurement_time() const { return measurement_time != 0.0; }
    bool has_width() const { return width != 0; }
    bool has_height() const { return height != 0; }
    size_t calculate_size() const {
      size_t size = 0;
      if (has_frame_id()) { size += varint_size(make_tag(2, LENGTH_DELIMITED)) + string_size(frame_id); }
      if (has_is_dense()) { size += varint_size(make_tag(3, VARINT)) + varint_size(is_dense ? 1 : 0); }
      if (has_points()) {
        for (const auto& point : points) {
          size_t point_size = point.calculate_size();
          size += varint_size(make_tag(4, LENGTH_DELIMITED)) + varint_size(point_size) + point_size;
        }
      }
      if (has_measurement_time()) { size += varint_size(make_tag(5, FIXED64)) + 8; }
      if (has_width()) { size += varint_size(make_tag(6, VARINT)) + varint_size(width); }
      if (has_height()) { size += varint_size(make_tag(7, VARINT)) + varint_size(height); }
      return size;
    }
    char* serialize(char* ptr) const {
      if (has_frame_id()) {
        ptr = encode_varint(ptr, make_tag(2, LENGTH_DELIMITED));
        ptr = encode_string(ptr, frame_id);
      }
      if (has_is_dense()) {
        ptr = encode_varint(ptr, make_tag(3, VARINT));
        ptr = encode_varint(ptr, is_dense ? 1 : 0);
      }
      if (has_points()) {
        for (const auto& point : points) {
          size_t point_size = point.calculate_size();
          ptr = encode_varint(ptr, make_tag(4, LENGTH_DELIMITED));
          ptr = encode_varint(ptr, point_size);
          ptr = point.serialize(ptr);
        }
      }
      if (has_measurement_time()) {
        ptr = encode_varint(ptr, make_tag(5, FIXED64));
        ptr = encode_double(ptr, measurement_time);
      }
      if (has_width()) {
        ptr = encode_varint(ptr, make_tag(6, VARINT));
        ptr = encode_varint(ptr, width);
      }
      if (has_height()) {
        ptr = encode_varint(ptr, make_tag(7, VARINT));
        ptr = encode_varint(ptr, height);
      }
      return ptr;
    }
    // 高性能序列化到用户提供的buffer
    size_t serialize_to_buffer(char* buffer) const {
      char* end_ptr = serialize(buffer);
      return end_ptr - buffer;
    }
    // 获取序列化所需的buffer大小
    size_t get_serialized_size() const { return calculate_size(); }
  };
  // 反序列化相关函数
  static const char* decode_varint(const char* ptr, const char* end, uint64_t& value) {
    value = 0;
    int shift = 0;
    while (ptr < end) {
      char byte = *ptr++;
      value |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) { return ptr; }
      shift += 7;
      if (shift >= 64) return nullptr; // overflow
    }
    return nullptr; // incomplete varint
  }
  static const char* decode_fixed32(const char* ptr, const char* end, uint32_t& value) {
    if (ptr + 4 > end) return nullptr;
    memcpy(&value, ptr, sizeof(uint32_t));
    return ptr + 4;
  }
  static const char* decode_fixed64(const char* ptr, const char* end, uint64_t& value) {
    if (ptr + 8 > end) return nullptr;
    memcpy(&value, ptr, sizeof(uint64_t));
    return ptr + 8;
  }
  static const char* decode_float(const char* ptr, const char* end, float& value) {
    uint32_t bits;
    ptr = decode_fixed32(ptr, end, bits);
    if (ptr) { memcpy(&value, &bits, sizeof(float)); }
    return ptr;
  }
  static const char* decode_double(const char* ptr, const char* end, double& value) {
    uint64_t bits;
    ptr = decode_fixed64(ptr, end, bits);
    if (ptr) { memcpy(&value, &bits, sizeof(double)); }
    return ptr;
  }
  static const char* decode_string(const char* ptr, const char* end, std::string& str) {
    uint64_t length;
    ptr = decode_varint(ptr, end, length);
    if (!ptr || ptr + length > end) return nullptr;
    str.assign(reinterpret_cast<const char*>(ptr), length);
    return ptr + length;
  }
  static bool deserialize_point(const char* data, size_t size, PointXYZIT& point) {
    const char* ptr = data;
    const char* end = data + size;
    while (ptr < end) {
      uint64_t tag;
      ptr = decode_varint(ptr, end, tag);
      if (!ptr) return false;
      uint32_t field_number = tag >> 3;
      WireType wire_type = static_cast<WireType>(tag & 0x7);
      switch (field_number) {
        case 1: // x
          if (wire_type != FIXED32) return false;
          ptr = decode_float(ptr, end, point.x);
          break;
        case 2: // y
          if (wire_type != FIXED32) return false;
          ptr = decode_float(ptr, end, point.y);
          break;
        case 3: // z
          if (wire_type != FIXED32) return false;
          ptr = decode_float(ptr, end, point.z);
          break;
        case 4: // intensity
          if (wire_type != VARINT) return false;
          {
            uint64_t value;
            ptr = decode_varint(ptr, end, value);
            point.intensity = static_cast<uint32_t>(value);
          }
          break;
        case 5: // timestamp
          if (wire_type != VARINT) return false;
          ptr = decode_varint(ptr, end, point.timestamp);
          break;
        default:
          // Skip unknown field
          switch (wire_type) {
            case VARINT: {
              uint64_t value;
              ptr = decode_varint(ptr, end, value);
            } break;
            case FIXED32:
              ptr += 4;
              break;
            case FIXED64:
              ptr += 8;
              break;
            case LENGTH_DELIMITED: {
              uint64_t length;
              ptr = decode_varint(ptr, end, length);
              if (ptr) ptr += length;
            } break;
            default:
              return false;
          }
          break;
      }
      if (!ptr) return false;
    }
    return true;
  }
  static bool deserialize_pointcloud(const char* data, size_t size, PointCloud& cloud) {
    const char* ptr = data;
    const char* end = data + size;
    while (ptr < end) {
      uint64_t tag;
      ptr = decode_varint(ptr, end, tag);
      if (!ptr) return false;
      uint32_t field_number = tag >> 3;
      WireType wire_type = static_cast<WireType>(tag & 0x7);
      switch (field_number) {
        case 2: // frame_id
          if (wire_type != LENGTH_DELIMITED) return false;
          ptr = decode_string(ptr, end, cloud.frame_id);
          break;
        case 3: // is_dense
          if (wire_type != VARINT) return false;
          {
            uint64_t value;
            ptr = decode_varint(ptr, end, value);
            cloud.is_dense = (value != 0);
          }
          break;
        case 4: // point
          if (wire_type != LENGTH_DELIMITED) return false;
          {
            uint64_t length;
            ptr = decode_varint(ptr, end, length);
            if (!ptr || ptr + length > end) return false;
            PointXYZIT point;
            if (!deserialize_point(ptr, length, point)) return false;
            cloud.points.push_back(point);
            ptr += length;
          }
          break;
        case 5: // measurement_time
          if (wire_type != FIXED64) return false;
          ptr = decode_double(ptr, end, cloud.measurement_time);
          break;
        case 6: // width
          if (wire_type != VARINT) return false;
          {
            uint64_t value;
            ptr = decode_varint(ptr, end, value);
            cloud.width = static_cast<uint32_t>(value);
          }
          break;
        case 7: // height
          if (wire_type != VARINT) return false;
          {
            uint64_t value;
            ptr = decode_varint(ptr, end, value);
            cloud.height = static_cast<uint32_t>(value);
          }
          break;
        default:
          // Skip unknown field
          switch (wire_type) {
            case VARINT: {
              uint64_t value;
              ptr = decode_varint(ptr, end, value);
            } break;
            case FIXED32:
              ptr += 4;
              break;
            case FIXED64:
              ptr += 8;
              break;
            case LENGTH_DELIMITED: {
              uint64_t length;
              ptr = decode_varint(ptr, end, length);
              if (ptr) ptr += length;
            } break;
            default:
              return false;
          }
          break;
      }
      if (!ptr) return false;
    }
    return true;
  }
};
} // namespace sdproto

// https://docs.google.com/document/d/1Z_mbEbwSsnO5BYBAftTytDHQwRwPIvuPWt10lGZwerg/edit?pli=1&tab=t.0
#include <cmath> // For std::isnan
#include <cstdint>
#include <cstring> // For memcpy
#include <iostream>
#include <limits> // For std::numeric_limits
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace sdproto_opt {
#pragma push
#pragma pack(1)
struct PointXYZIT {
  double x{std::numeric_limits<float>::quiet_NaN()}, y{std::numeric_limits<float>::quiet_NaN()}, z{std::numeric_limits<float>::quiet_NaN()};
  double intensity{0};
  uint64_t timestamp{0};
};

struct PointCloud {
  std::optional<std::string> frame_id;
  std::optional<bool> is_dense;
  std::vector<PointXYZIT> point;
  std::optional<double> measurement_time;
  std::optional<uint32_t> width;
  std::optional<uint32_t> height;
};
// Protobuf Wire Types
enum class WireType : char {
  Varint = 0,
  Bit64 = 1,
  LengthDelimited = 2,
  Bit32 = 5,
};
// --- Varint 编码 ---
inline size_t SizeofVarint(uint64_t value) {
  if (value < (1ULL << 7)) return 1;
  if (value < (1ULL << 14)) return 2;
  if (value < (1ULL << 21)) return 3;
  if (value < (1ULL << 28)) return 4;
  if (value < (1ULL << 35)) return 5;
  if (value < (1ULL << 42)) return 6;
  if (value < (1ULL << 49)) return 7;
  if (value < (1ULL << 56)) return 8;
  if (value < (1ULL << 63)) return 9;
  return 10;
}

inline char* EncodeVarint(char* target, uint64_t value) {
  while (value >= 0x80) {
    *target++ = (static_cast<char>(value) & 0x7F) | 0x80;
    value >>= 7;
  }
  *target++ = static_cast<char>(value);
  return target;
}

inline bool DecodeVarint(const char*& data, const char* end, uint64_t& value) {
  const char* ptr = data;
  uint64_t result = 0;
  int shift = 0;
  while (ptr < end && shift <= 63) {
    char byte = *ptr++;
    result |= (static_cast<uint64_t>(byte & 0x7F)) << shift;
    if ((byte & 0x80) == 0) {
      data = ptr;
      value = result;
      return true;
    }
    shift += 7;
  }
  return false;
}

// --- Tag 生成 ---
inline uint32_t MakeTag(uint32_t field_number, WireType type) { return (field_number << 3) | static_cast<char>(type); }

// --- 基础类型序列化 ---
// 固定32位 (float, fixed32, sfixed32)
inline size_t SizeofFixed32(uint32_t field_number) { return SizeofVarint(MakeTag(field_number, WireType::Bit32)) + 4; }
inline char* SerializeFixed32(char* target, uint32_t field_number, void* value) {
  target = EncodeVarint(target, MakeTag(field_number, WireType::Bit32));
  std::memcpy(target, value, 4);
  return target + 4;
}
inline bool DecodeFixed32(const char*& data, const char* end, void* value) {
  if (data + 4 > end) return false;
  std::memcpy(value, data, 4);
  data += 4;
  return true;
}
// 固定64位 (double, fixed64, sfixed64)
inline size_t SizeofFixed64(uint32_t field_number) { return SizeofVarint(MakeTag(field_number, WireType::Bit64)) + 8; }
inline char* SerializeFixed64(char* target, uint32_t field_number, void* value) {
  target = EncodeVarint(target, MakeTag(field_number, WireType::Bit64));
  std::memcpy(target, value, 8);
  return target + 8;
}
inline bool DecodeFixed64(const char*& data, const char* end, void* value) {
  if (data + 8 > end) return false;
  std::memcpy(value, data, 8);
  data += 8;
  return true;
}
// 变长 (string, bytes, sub-message)
inline size_t SizeofLengthDelimited(uint32_t field_number, size_t len) { return SizeofVarint(MakeTag(field_number, WireType::LengthDelimited)) + SizeofVarint(len) + len; }
inline bool DecodeLengthDelimited(const char*& data, const char* end, std::string_view& view) {
  uint64_t length;
  if (!DecodeVarint(data, end, length)) return false;
  if (data + length > end) return false;
  view = std::string_view(reinterpret_cast<const char*>(data), static_cast<size_t>(length));
  data += length;
  return true;
}
// 跳过未知字段
inline bool SkipField(const char*& data, const char* end, WireType wire_type) {
  switch (wire_type) {
    case WireType::Varint: {
      uint64_t dummy;
      return DecodeVarint(data, end, dummy);
    }
    case WireType::Bit64: {
      if (data + 8 > end) return false;
      data += 8;
      return true;
    }
    case WireType::Bit32: {
      if (data + 4 > end) return false;
      data += 4;
      return true;
    }
    case WireType::LengthDelimited: {
      uint64_t length;
      if (!DecodeVarint(data, end, length)) return false;
      if (data + length > end) return false;
      data += length;
      return true;
    }
    default:
      return false;
  }
}
// --- PointXYZIT Serializer ---
namespace PointXYZITSerializer {
size_t GetSerializedSize(const PointXYZIT& point) {
  size_t total_size = 0;
  // 只有当值不等于默认值时，才计算其大小
  total_size += SizeofFixed64(1);
  total_size += SizeofFixed64(2);
  total_size += SizeofFixed64(3);
  total_size += SizeofFixed64(4);
  total_size += SizeofFixed64(5);
  return total_size;
}
// 注意：此函数不进行边界检查，调用者需保证buffer足够大
char* Serialize(const PointXYZIT& point, char* target) {
  target = SerializeFixed64(target, 1, (void*)&point.x);
  target = SerializeFixed64(target, 2, (void*)&point.y);
  target = SerializeFixed64(target, 3, (void*)&point.z);
  target = SerializeFixed64(target, 4, (void*)&point.intensity);
  target = SerializeFixed64(target, 5, (void*)&point.timestamp);
  return target;
}
bool Deserialize(PointXYZIT& point, const char* buffer, size_t size) {
  point = {}; // 重置为默认值
  const char* data = buffer;
  const char* end = buffer + size;
  while (data < end) {
    uint64_t tag_val;
    if (!DecodeVarint(data, end, tag_val)) return false;
    uint32_t field_number = static_cast<uint32_t>(tag_val >> 3);
    WireType wire_type = static_cast<WireType>(tag_val & 0x7);
    switch (field_number) {
      case 1: { // x
        DecodeFixed64(data, end, &point.x);
        break;
      }
      case 2: { // y
        DecodeFixed64(data, end, &point.y);
        break;
      }
      case 3: { // z
        DecodeFixed64(data, end, &point.z);
        break;
      }
      case 4: { // intensity
        DecodeFixed64(data, end, &point.intensity);
        break;
      }
      case 5: { // timestamp
        DecodeFixed64(data, end, &point.timestamp);
        break;
      }
      default:
        if (!SkipField(data, end, wire_type)) return false;
        break;
    }
  }
  return data == end;
}
} // namespace PointXYZITSerializer
// --- PointCloud Serializer ---
namespace PointCloudSerializer {
size_t GetSerializedSize(const PointCloud& cloud) {
  size_t total_size = 0;
  if (cloud.frame_id) total_size += SizeofLengthDelimited(2, cloud.frame_id->length());
  if (cloud.is_dense) total_size += SizeofVarint(MakeTag(3, WireType::Varint)) + 1; // bool is always 1 byte varint
  for (const auto& p : cloud.point) {
    const size_t point_size = PointXYZITSerializer::GetSerializedSize(p);
    total_size += SizeofLengthDelimited(4, point_size);
  }
  if (cloud.measurement_time) total_size += SizeofFixed64(5);
  if (cloud.width) total_size += SizeofVarint(MakeTag(6, WireType::Varint)) + SizeofVarint(*cloud.width);
  if (cloud.height) total_size += SizeofVarint(MakeTag(7, WireType::Varint)) + SizeofVarint(*cloud.height);
  return total_size;
}
bool Serialize(const PointCloud& cloud, char* buffer, size_t size) {
  char* target = buffer;
  const size_t expected_size = GetSerializedSize(cloud);
  if (size < expected_size) return false; // 缓冲区大小不足
  if (cloud.frame_id) {
    target = EncodeVarint(target, MakeTag(2, WireType::LengthDelimited));
    target = EncodeVarint(target, cloud.frame_id->length());
    std::memcpy(target, cloud.frame_id->data(), cloud.frame_id->length());
    target += cloud.frame_id->length();
  }
  if (cloud.is_dense) {
    target = EncodeVarint(target, MakeTag(3, WireType::Varint));
    target = EncodeVarint(target, *cloud.is_dense ? 1 : 0);
  }
  for (const auto& p : cloud.point) {
    const size_t point_size = PointXYZITSerializer::GetSerializedSize(p);
    target = EncodeVarint(target, MakeTag(4, WireType::LengthDelimited));
    target = EncodeVarint(target, point_size);
    target = PointXYZITSerializer::Serialize(p, target);
  }
  if (cloud.measurement_time) {
    uint64_t u64_val;
    std::memcpy(&u64_val, &(*cloud.measurement_time), 8);
    target = SerializeFixed64(target, 5, (void*)&u64_val);
  }
  if (cloud.width) {
    target = EncodeVarint(target, MakeTag(6, WireType::Varint));
    target = EncodeVarint(target, *cloud.width);
  }
  if (cloud.height) {
    target = EncodeVarint(target, MakeTag(7, WireType::Varint));
    target = EncodeVarint(target, *cloud.height);
  }
  return (target - buffer) == expected_size;
}
bool Deserialize(PointCloud& cloud, const char* buffer, size_t size) {
  cloud = {};
  const char* data = buffer;
  const char* end = buffer + size;
  while (data < end) {
    uint64_t tag_val;
    if (!DecodeVarint(data, end, tag_val)) return false;
    uint32_t field_number = static_cast<uint32_t>(tag_val >> 3);
    WireType wire_type = static_cast<WireType>(tag_val & 0x7);
    switch (field_number) {
      case 2: { // frame_id
        if (wire_type != WireType::LengthDelimited) return false;
        std::string_view view;
        if (!DecodeLengthDelimited(data, end, view)) return false;
        cloud.frame_id = std::string(view);
        break;
      }
      case 3: { // is_dense
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        cloud.is_dense = (val != 0);
        break;
      }
      case 4: { // point
        if (wire_type != WireType::LengthDelimited) return false;
        uint64_t length;
        const char* sub_start = data;
        if (!DecodeVarint(data, end, length)) return false;
        if (data + length > end) return false;
        PointXYZIT p;
        if (!PointXYZITSerializer::Deserialize(p, data, length)) return false;
        cloud.point.push_back(std::move(p));
        data += length;
        break;
      }
      case 5: { // measurement_time
        if (wire_type != WireType::Bit64) return false;
        uint64_t u64_val;
        if (!DecodeFixed64(data, end, &u64_val)) return false;
        double d_val;
        std::memcpy(&d_val, &u64_val, 8);
        cloud.measurement_time = d_val;
        break;
      }
      case 6: { // width
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        cloud.width = static_cast<uint32_t>(val);
        break;
      }
      case 7: { // height
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        cloud.height = static_cast<uint32_t>(val);
        break;
      }
      default:
        if (!SkipField(data, end, wire_type)) return false;
        break;
    }
  }
  return data == end;
}
#pragma pop
} // namespace PointCloudSerializer
} // namespace sdproto_opt

/*namespace sdproto_opt {
struct PointXYZIT {
  float x = std::numeric_limits<float>::quiet_NaN();
  float y = std::numeric_limits<float>::quiet_NaN();
  float z = std::numeric_limits<float>::quiet_NaN();
  uint32_t intensity = 0;
  uint64_t timestamp = 0;
};
struct PointCloud {
  std::optional<std::string> frame_id;
  std::optional<bool> is_dense;
  std::vector<PointXYZIT> point;
  std::optional<double> measurement_time;
  std::optional<uint32_t> width;
  std::optional<uint32_t> height;
};
// Protobuf Wire Types
enum class WireType : char {
  Varint = 0,
  Bit64 = 1,
  LengthDelimited = 2,
  Bit32 = 5,
};
// --- Varint 编码 ---
inline size_t SizeofVarint(uint64_t value) {
  if (value < (1ULL << 7)) return 1;
  if (value < (1ULL << 14)) return 2;
  if (value < (1ULL << 21)) return 3;
  if (value < (1ULL << 28)) return 4;
  if (value < (1ULL << 35)) return 5;
  if (value < (1ULL << 42)) return 6;
  if (value < (1ULL << 49)) return 7;
  if (value < (1ULL << 56)) return 8;
  if (value < (1ULL << 63)) return 9;
  return 10;
}
inline char* EncodeVarint(char* target, uint64_t value) {
  while (value >= 0x80) {
    *target++ = (static_cast<char>(value) & 0x7F) | 0x80;
    value >>= 7;
  }
  *target++ = static_cast<char>(value);
  return target;
}
inline bool DecodeVarint(const char*& data, const char* end, uint64_t& value) {
  const char* ptr = data;
  uint64_t result = 0;
  int shift = 0;
  while (ptr < end && shift <= 63) {
    char byte = *ptr++;
    result |= (static_cast<uint64_t>(byte & 0x7F)) << shift;
    if ((byte & 0x80) == 0) {
      data = ptr;
      value = result;
      return true;
    }
    shift += 7;
  }
  return false;
}
// --- Tag 生成 ---
inline uint32_t MakeTag(uint32_t field_number, WireType type) { return (field_number << 3) | static_cast<char>(type); }
// --- 基础类型序列化 ---
// 固定32位 (float, fixed32, sfixed32)
inline size_t SizeofFixed32(uint32_t field_number) { return SizeofVarint(MakeTag(field_number, WireType::Bit32)) + 4; }
inline char* SerializeFixed32(char* target, uint32_t field_number, uint32_t value) {
  target = EncodeVarint(target, MakeTag(field_number, WireType::Bit32));
  std::memcpy(target, &value, 4);
  return target + 4;
}
inline bool DecodeFixed32(const char*& data, const char* end, uint32_t& value) {
  if (data + 4 > end) return false;
  std::memcpy(&value, data, 4);
  data += 4;
  return true;
}
// 固定64位 (double, fixed64, sfixed64)
inline size_t SizeofFixed64(uint32_t field_number) { return SizeofVarint(MakeTag(field_number, WireType::Bit64)) + 8; }
inline char* SerializeFixed64(char* target, uint32_t field_number, uint64_t value) {
  target = EncodeVarint(target, MakeTag(field_number, WireType::Bit64));
  std::memcpy(target, &value, 8);
  return target + 8;
}
inline bool DecodeFixed64(const char*& data, const char* end, uint64_t& value) {
  if (data + 8 > end) return false;
  std::memcpy(&value, data, 8);
  data += 8;
  return true;
}
// 变长 (string, bytes, sub-message)
inline size_t SizeofLengthDelimited(uint32_t field_number, size_t len) { return SizeofVarint(MakeTag(field_number, WireType::LengthDelimited)) + SizeofVarint(len) + len; }
inline bool DecodeLengthDelimited(const char*& data, const char* end, std::string_view& view) {
  uint64_t length;
  if (!DecodeVarint(data, end, length)) return false;
  if (data + length > end) return false;
  view = std::string_view(reinterpret_cast<const char*>(data), static_cast<size_t>(length));
  data += length;
  return true;
}
// 跳过未知字段
inline bool SkipField(const char*& data, const char* end, WireType wire_type) {
  switch (wire_type) {
    case WireType::Varint: {
      uint64_t dummy;
      return DecodeVarint(data, end, dummy);
    }
    case WireType::Bit64: {
      if (data + 8 > end) return false;
      data += 8;
      return true;
    }
    case WireType::Bit32: {
      if (data + 4 > end) return false;
      data += 4;
      return true;
    }
    case WireType::LengthDelimited: {
      uint64_t length;
      if (!DecodeVarint(data, end, length)) return false;
      if (data + length > end) return false;
      data += length;
      return true;
    }
    default:
      return false;
  }
}
// --- PointXYZIT Serializer ---
namespace PointXYZITSerializer {
size_t GetSerializedSize(const PointXYZIT& point) {
  size_t total_size = 0;
  // 只有当值不等于默认值时，才计算其大小
  if (!std::isnan(point.x)) total_size += SizeofFixed32(1);
  if (!std::isnan(point.y)) total_size += SizeofFixed32(2);
  if (!std::isnan(point.z)) total_size += SizeofFixed32(3);
  if (point.intensity != 0) total_size += SizeofVarint(MakeTag(4, WireType::Varint)) + SizeofVarint(point.intensity);
  if (point.timestamp != 0) total_size += SizeofVarint(MakeTag(5, WireType::Varint)) + SizeofVarint(point.timestamp);
  return total_size;
}
// 注意：此函数不进行边界检查，调用者需保证buffer足够大
char* Serialize(const PointXYZIT& point, char* target) {
  uint32_t u32_val;
  if (!std::isnan(point.x)) {
    std::memcpy(&u32_val, &point.x, 4);
    target = SerializeFixed32(target, 1, u32_val);
  }
  if (!std::isnan(point.y)) {
    std::memcpy(&u32_val, &point.y, 4);
    target = SerializeFixed32(target, 2, u32_val);
  }
  if (!std::isnan(point.z)) {
    std::memcpy(&u32_val, &point.z, 4);
    target = SerializeFixed32(target, 3, u32_val);
  }
  if (point.intensity != 0) {
    target = EncodeVarint(target, MakeTag(4, WireType::Varint));
    target = EncodeVarint(target, point.intensity);
  }
  if (point.timestamp != 0) {
    target = EncodeVarint(target, MakeTag(5, WireType::Varint));
    target = EncodeVarint(target, point.timestamp);
  }
  return target;
}
bool Deserialize(PointXYZIT& point, const char* buffer, size_t size) {
  point = {}; // 重置为默认值
  const char* data = buffer;
  const char* end = buffer + size;
  while (data < end) {
    uint64_t tag_val;
    if (!DecodeVarint(data, end, tag_val)) return false;
    uint32_t field_number = static_cast<uint32_t>(tag_val >> 3);
    WireType wire_type = static_cast<WireType>(tag_val & 0x7);
    switch (field_number) {
      case 1: { // x
        if (wire_type != WireType::Bit32) return false;
        uint32_t u32_val;
        if (!DecodeFixed32(data, end, u32_val)) return false;
        std::memcpy(&point.x, &u32_val, 4);
        break;
      }
      case 2: { // y
        if (wire_type != WireType::Bit32) return false;
        uint32_t u32_val;
        if (!DecodeFixed32(data, end, u32_val)) return false;
        std::memcpy(&point.y, &u32_val, 4);
        break;
      }
      case 3: { // z
        if (wire_type != WireType::Bit32) return false;
        uint32_t u32_val;
        if (!DecodeFixed32(data, end, u32_val)) return false;
        std::memcpy(&point.z, &u32_val, 4);
        break;
      }
      case 4: { // intensity
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        point.intensity = static_cast<uint32_t>(val);
        break;
      }
      case 5: { // timestamp
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        point.timestamp = val;
        break;
      }
      default:
        if (!SkipField(data, end, wire_type)) return false;
        break;
    }
  }
  return data == end;
}
} // namespace PointXYZITSerializer
// --- PointCloud Serializer ---
namespace PointCloudSerializer {
size_t GetSerializedSize(const PointCloud& cloud) {
  size_t total_size = 0;
  if (cloud.frame_id) total_size += SizeofLengthDelimited(2, cloud.frame_id->length());
  if (cloud.is_dense) total_size += SizeofVarint(MakeTag(3, WireType::Varint)) + 1; // bool is always 1 byte varint
  for (const auto& p : cloud.point) {
    const size_t point_size = PointXYZITSerializer::GetSerializedSize(p);
    total_size += SizeofLengthDelimited(4, point_size);
  }
  if (cloud.measurement_time) total_size += SizeofFixed64(5);
  if (cloud.width) total_size += SizeofVarint(MakeTag(6, WireType::Varint)) + SizeofVarint(*cloud.width);
  if (cloud.height) total_size += SizeofVarint(MakeTag(7, WireType::Varint)) + SizeofVarint(*cloud.height);
  return total_size;
}
bool Serialize(const PointCloud& cloud, char* buffer, size_t size) {
  char* target = buffer;
  const size_t expected_size = GetSerializedSize(cloud);
  if (size < expected_size) return false; // 缓冲区大小不足
  if (cloud.frame_id) {
    target = EncodeVarint(target, MakeTag(2, WireType::LengthDelimited));
    target = EncodeVarint(target, cloud.frame_id->length());
    std::memcpy(target, cloud.frame_id->data(), cloud.frame_id->length());
    target += cloud.frame_id->length();
  }
  if (cloud.is_dense) {
    target = EncodeVarint(target, MakeTag(3, WireType::Varint));
    target = EncodeVarint(target, *cloud.is_dense ? 1 : 0);
  }
  for (const auto& p : cloud.point) {
    const size_t point_size = PointXYZITSerializer::GetSerializedSize(p);
    target = EncodeVarint(target, MakeTag(4, WireType::LengthDelimited));
    target = EncodeVarint(target, point_size);
    target = PointXYZITSerializer::Serialize(p, target);
  }
  if (cloud.measurement_time) {
    uint64_t u64_val;
    std::memcpy(&u64_val, &(*cloud.measurement_time), 8);
    target = SerializeFixed64(target, 5, u64_val);
  }
  if (cloud.width) {
    target = EncodeVarint(target, MakeTag(6, WireType::Varint));
    target = EncodeVarint(target, *cloud.width);
  }
  if (cloud.height) {
    target = EncodeVarint(target, MakeTag(7, WireType::Varint));
    target = EncodeVarint(target, *cloud.height);
  }
  return (target - buffer) == expected_size;
}
bool Deserialize(PointCloud& cloud, const char* buffer, size_t size) {
  cloud = {};
  const char* data = buffer;
  const char* end = buffer + size;
  while (data < end) {
    uint64_t tag_val;
    if (!DecodeVarint(data, end, tag_val)) return false;
    uint32_t field_number = static_cast<uint32_t>(tag_val >> 3);
    WireType wire_type = static_cast<WireType>(tag_val & 0x7);

    switch (field_number) {
      case 2: { // frame_id
        if (wire_type != WireType::LengthDelimited) return false;
        std::string_view view;
        if (!DecodeLengthDelimited(data, end, view)) return false;
        cloud.frame_id = std::string(view);
        break;
      }
      case 3: { // is_dense
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        cloud.is_dense = (val != 0);
        break;
      }
      case 4: { // point
        if (wire_type != WireType::LengthDelimited) return false;
        uint64_t length;
        const char* sub_start = data;
        if (!DecodeVarint(data, end, length)) return false;
        if (data + length > end) return false;
        PointXYZIT p;
        if (!PointXYZITSerializer::Deserialize(p, data, length)) return false;
        cloud.point.push_back(std::move(p));
        data += length;
        break;
      }
      case 5: { // measurement_time
        if (wire_type != WireType::Bit64) return false;
        uint64_t u64_val;
        if (!DecodeFixed64(data, end, u64_val)) return false;
        double d_val;
        std::memcpy(&d_val, &u64_val, 8);
        cloud.measurement_time = d_val;
        break;
      }
      case 6: { // width
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        cloud.width = static_cast<uint32_t>(val);
        break;
      }
      case 7: { // height
        if (wire_type != WireType::Varint) return false;
        uint64_t val;
        if (!DecodeVarint(data, end, val)) return false;
        cloud.height = static_cast<uint32_t>(val);
        break;
      }
      default:
        if (!SkipField(data, end, wire_type)) return false;
        break;
    }
  }
  return data == end;
}
} // namespace PointCloudSerializer
} // namespace sdproto_opt*/
