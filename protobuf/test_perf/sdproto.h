#pragma once

// https://claude.ai/chat/3f94f4c4-f43f-4adf-92fb-2b788ceb5b20
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace sdproto {
// Wire types for protobuf encoding
enum WireType { VARINT = 0, FIXED64 = 1, LENGTH_DELIMITED = 2, FIXED32 = 5 };

class ProtoBuffer {
private:
  std::vector<char>* buffer_;
  size_t read_pos_;
  bool external_buffer_;

public:
  ProtoBuffer() : buffer_(new std::vector<char>()), read_pos_(0), external_buffer_(false) {}

  ProtoBuffer(std::vector<char>& external_buf) : buffer_(&external_buf), read_pos_(0), external_buffer_(true) {}

  ~ProtoBuffer() {
    if (!external_buffer_) { delete buffer_; }
  }

  // Encode varint (variable length integer)
  void encodeVarint(uint64_t value) {
    while (value >= 0x80) {
      buffer_->push_back((value & 0xFF) | 0x80);
      value >>= 7;
    }
    buffer_->push_back(value & 0xFF);
  }

  // Decode varint
  uint64_t decodeVarint() {
    uint64_t result = 0;
    int shift = 0;
    while (read_pos_ < buffer_->size()) {
      char byte = (*buffer_)[read_pos_++];
      result |= (uint64_t)(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) { break; }
      shift += 7;
      if (shift >= 64) { throw std::runtime_error("Varint too long"); }
    }
    return result;
  }

  // Encode field key (field_number << 3 | wire_type)
  void encodeFieldKey(int field_number, WireType wire_type) { encodeVarint((field_number << 3) | wire_type); }

  // Decode field key
  std::pair<int, WireType> decodeFieldKey() {
    uint64_t key = decodeVarint();
    int field_number = key >> 3;
    WireType wire_type = static_cast<WireType>(key & 0x7);
    return {field_number, wire_type};
  }

  // Encode string/bytes
  void encodeString(const std::string& str) {
    encodeVarint(str.length());
    buffer_->insert(buffer_->end(), str.begin(), str.end());
  }

  // Decode string/bytes
  std::string decodeString() {
    uint64_t length = decodeVarint();
    if (read_pos_ + length > buffer_->size()) { throw std::runtime_error("String extends beyond buffer"); }
    std::string result(buffer_->begin() + read_pos_, buffer_->begin() + read_pos_ + length);
    read_pos_ += length;
    return result;
  }

  // Encode fixed32
  void encodeFixed32(uint32_t value) {
    buffer_->push_back(value & 0xFF);
    buffer_->push_back((value >> 8) & 0xFF);
    buffer_->push_back((value >> 16) & 0xFF);
    buffer_->push_back((value >> 24) & 0xFF);
  }

  // Decode fixed32
  uint32_t decodeFixed32() {
    if (read_pos_ + 4 > buffer_->size()) { throw std::runtime_error("Fixed32 extends beyond buffer"); }
    uint32_t result = (*buffer_)[read_pos_] | ((*buffer_)[read_pos_ + 1] << 8) | ((*buffer_)[read_pos_ + 2] << 16) | ((*buffer_)[read_pos_ + 3] << 24);
    read_pos_ += 4;
    return result;
  }

  // Encode fixed64
  void encodeFixed64(uint64_t value) {
    for (int i = 0; i < 8; i++) { buffer_->push_back((value >> (i * 8)) & 0xFF); }
  }

  // Decode fixed64
  uint64_t decodeFixed64() {
    if (read_pos_ + 8 > buffer_->size()) { throw std::runtime_error("Fixed64 extends beyond buffer"); }
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) { result |= (uint64_t)(*buffer_)[read_pos_ + i] << (i * 8); }
    read_pos_ += 8;
    return result;
  }

  // Encode length-delimited data
  void encodeLengthDelimited(const std::vector<char>& data) {
    encodeVarint(data.size());
    buffer_->insert(buffer_->end(), data.begin(), data.end());
  }

  // Decode length-delimited data
  std::vector<char> decodeLengthDelimited() {
    uint64_t length = decodeVarint();
    if (read_pos_ + length > buffer_->size()) { throw std::runtime_error("Length delimited data extends beyond buffer"); }
    std::vector<char> result(buffer_->begin() + read_pos_, buffer_->begin() + read_pos_ + length);
    read_pos_ += length;
    return result;
  }

  // Get buffer data
  const std::vector<char>& getData() const { return *buffer_; }

  // Set data for reading
  void setDataForReading(const std::vector<char>& data) {
    if (!external_buffer_) {
      *buffer_ = data;
    } else {
      throw std::runtime_error("Cannot set data on external buffer");
    }
    read_pos_ = 0;
  }

  // Check if at end of buffer
  bool atEnd() const { return read_pos_ >= buffer_->size(); }

  // Reset read position
  void resetReadPos() { read_pos_ = 0; }

  // Clear buffer
  void clear() {
    buffer_->clear();
    read_pos_ = 0;
  }

  // Get current size
  size_t size() const { return buffer_->size(); }
};

// PointXYZIT message implementation
class PointXYZIT {
private:
  float x_;
  float y_;
  float z_;
  uint32_t intensity_;
  uint64_t timestamp_;

  // Field presence flags
  bool has_x_ = false;
  bool has_y_ = false;
  bool has_z_ = false;
  bool has_intensity_ = false;
  bool has_timestamp_ = false;

public:
  PointXYZIT() {
    // Set default values
    x_ = std::numeric_limits<float>::quiet_NaN();
    y_ = std::numeric_limits<float>::quiet_NaN();
    z_ = std::numeric_limits<float>::quiet_NaN();
    intensity_ = 0;
    timestamp_ = 0;
  }

  // Setters
  void set_x(float value) {
    x_ = value;
    has_x_ = true;
  }
  void set_y(float value) {
    y_ = value;
    has_y_ = true;
  }
  void set_z(float value) {
    z_ = value;
    has_z_ = true;
  }
  void set_intensity(uint32_t value) {
    intensity_ = value;
    has_intensity_ = true;
  }
  void set_timestamp(uint64_t value) {
    timestamp_ = value;
    has_timestamp_ = true;
  }

  // Getters
  float x() const { return x_; }
  float y() const { return y_; }
  float z() const { return z_; }
  uint32_t intensity() const { return intensity_; }
  uint64_t timestamp() const { return timestamp_; }

  // Has methods
  bool has_x() const { return has_x_; }
  bool has_y() const { return has_y_; }
  bool has_z() const { return has_z_; }
  bool has_intensity() const { return has_intensity_; }
  bool has_timestamp() const { return has_timestamp_; }

  // Serialize to external buffer
  void serializeTo(std::vector<char>& buffer) {
    ProtoBuffer proto_buffer(buffer);

    if (has_x_) {
      proto_buffer.encodeFieldKey(1, FIXED32);
      uint32_t x_bits;
      memcpy(&x_bits, &x_, sizeof(float));
      proto_buffer.encodeFixed32(x_bits);
    }

    if (has_y_) {
      proto_buffer.encodeFieldKey(2, FIXED32);
      uint32_t y_bits;
      memcpy(&y_bits, &y_, sizeof(float));
      proto_buffer.encodeFixed32(y_bits);
    }

    if (has_z_) {
      proto_buffer.encodeFieldKey(3, FIXED32);
      uint32_t z_bits;
      memcpy(&z_bits, &z_, sizeof(float));
      proto_buffer.encodeFixed32(z_bits);
    }

    if (has_intensity_) {
      proto_buffer.encodeFieldKey(4, VARINT);
      proto_buffer.encodeVarint(intensity_);
    }

    if (has_timestamp_) {
      proto_buffer.encodeFieldKey(5, VARINT);
      proto_buffer.encodeVarint(timestamp_);
    }
  }

  // Deserialize from external buffer
  void deserializeFrom(const std::vector<char>& buffer) {
    std::vector<char> buffer_copy = buffer;
    ProtoBuffer proto_buffer(buffer_copy);
    proto_buffer.resetReadPos();

    while (!proto_buffer.atEnd()) {
      auto [field_number, wire_type] = proto_buffer.decodeFieldKey();

      switch (field_number) {
        case 1: // x
          if (wire_type == FIXED32) {
            uint32_t bits = proto_buffer.decodeFixed32();
            memcpy(&x_, &bits, sizeof(float));
            has_x_ = true;
          }
          break;
        case 2: // y
          if (wire_type == FIXED32) {
            uint32_t bits = proto_buffer.decodeFixed32();
            memcpy(&y_, &bits, sizeof(float));
            has_y_ = true;
          }
          break;
        case 3: // z
          if (wire_type == FIXED32) {
            uint32_t bits = proto_buffer.decodeFixed32();
            memcpy(&z_, &bits, sizeof(float));
            has_z_ = true;
          }
          break;
        case 4: // intensity
          if (wire_type == VARINT) {
            intensity_ = static_cast<uint32_t>(proto_buffer.decodeVarint());
            has_intensity_ = true;
          }
          break;
        case 5: // timestamp
          if (wire_type == VARINT) {
            timestamp_ = proto_buffer.decodeVarint();
            has_timestamp_ = true;
          }
          break;
        default:
          // Skip unknown field
          skipField(proto_buffer, wire_type);
          break;
      }
    }
  }

private:
  void skipField(ProtoBuffer& buffer, WireType wire_type) {
    switch (wire_type) {
      case VARINT:
        buffer.decodeVarint();
        break;
      case FIXED64:
        buffer.decodeFixed64();
        break;
      case LENGTH_DELIMITED:
        buffer.decodeLengthDelimited();
        break;
      case FIXED32:
        buffer.decodeFixed32();
        break;
    }
  }
};

// PointCloud message implementation
class PointCloud {
private:
  std::string frame_id_;
  bool is_dense_;
  std::vector<PointXYZIT> points_;
  double measurement_time_;
  uint32_t width_;
  uint32_t height_;

  // Field presence flags
  bool has_frame_id_ = false;
  bool has_is_dense_ = false;
  bool has_measurement_time_ = false;
  bool has_width_ = false;
  bool has_height_ = false;

public:
  PointCloud() {
    is_dense_ = false;
    measurement_time_ = 0.0;
    width_ = 0;
    height_ = 0;
  }

  // Setters
  void set_frame_id(const std::string& value) {
    frame_id_ = value;
    has_frame_id_ = true;
  }
  void set_is_dense(bool value) {
    is_dense_ = value;
    has_is_dense_ = true;
  }
  void set_measurement_time(double value) {
    measurement_time_ = value;
    has_measurement_time_ = true;
  }
  void set_width(uint32_t value) {
    width_ = value;
    has_width_ = true;
  }
  void set_height(uint32_t value) {
    height_ = value;
    has_height_ = true;
  }

  // Point management
  void add_point(const PointXYZIT& point) { points_.push_back(point); }
  PointXYZIT* add_point() {
    points_.emplace_back();
    return &points_.back();
  }
  void clear_points() { points_.clear(); }

  // Getters
  const std::string& frame_id() const { return frame_id_; }
  bool is_dense() const { return is_dense_; }
  double measurement_time() const { return measurement_time_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  const std::vector<PointXYZIT>& points() const { return points_; }
  std::vector<PointXYZIT>& mutable_points() { return points_; }
  size_t point_size() const { return points_.size(); }
  const PointXYZIT& point(size_t index) const { return points_.at(index); }
  PointXYZIT& mutable_point(size_t index) { return points_.at(index); }

  // Has methods
  bool has_frame_id() const { return has_frame_id_; }
  bool has_is_dense() const { return has_is_dense_; }
  bool has_measurement_time() const { return has_measurement_time_; }
  bool has_width() const { return has_width_; }
  bool has_height() const { return has_height_; }

  // Serialize to external buffer
  void serializeTo(std::vector<char>& buffer) {
    buffer.clear();
    ProtoBuffer proto_buffer(buffer);

    if (has_frame_id_) {
      proto_buffer.encodeFieldKey(2, LENGTH_DELIMITED);
      proto_buffer.encodeString(frame_id_);
    }

    if (has_is_dense_) {
      proto_buffer.encodeFieldKey(3, VARINT);
      proto_buffer.encodeVarint(is_dense_ ? 1 : 0);
    }

    // Serialize repeated points
    for (const auto& point : points_) {
      proto_buffer.encodeFieldKey(4, LENGTH_DELIMITED);

      // Serialize point to temporary buffer
      std::vector<char> point_buffer;
      const_cast<PointXYZIT&>(point).serializeTo(point_buffer);
      proto_buffer.encodeLengthDelimited(point_buffer);
    }

    if (has_measurement_time_) {
      proto_buffer.encodeFieldKey(5, FIXED64);
      uint64_t time_bits;
      memcpy(&time_bits, &measurement_time_, sizeof(double));
      proto_buffer.encodeFixed64(time_bits);
    }

    if (has_width_) {
      proto_buffer.encodeFieldKey(6, VARINT);
      proto_buffer.encodeVarint(width_);
    }

    if (has_height_) {
      proto_buffer.encodeFieldKey(7, VARINT);
      proto_buffer.encodeVarint(height_);
    }
  }

  // Deserialize from external buffer
  void deserializeFrom(const std::vector<char>& buffer) {
    std::vector<char> buffer_copy = buffer;
    ProtoBuffer proto_buffer(buffer_copy);
    proto_buffer.resetReadPos();

    while (!proto_buffer.atEnd()) {
      auto [field_number, wire_type] = proto_buffer.decodeFieldKey();

      switch (field_number) {
        case 2: // frame_id
          if (wire_type == LENGTH_DELIMITED) {
            frame_id_ = proto_buffer.decodeString();
            has_frame_id_ = true;
          }
          break;
        case 3: // is_dense
          if (wire_type == VARINT) {
            is_dense_ = proto_buffer.decodeVarint() != 0;
            has_is_dense_ = true;
          }
          break;
        case 4: // point (repeated)
          if (wire_type == LENGTH_DELIMITED) {
            std::vector<char> point_data = proto_buffer.decodeLengthDelimited();
            PointXYZIT point;
            point.deserializeFrom(point_data);
            points_.push_back(point);
          }
          break;
        case 5: // measurement_time
          if (wire_type == FIXED64) {
            uint64_t bits = proto_buffer.decodeFixed64();
            memcpy(&measurement_time_, &bits, sizeof(double));
            has_measurement_time_ = true;
          }
          break;
        case 6: // width
          if (wire_type == VARINT) {
            width_ = static_cast<uint32_t>(proto_buffer.decodeVarint());
            has_width_ = true;
          }
          break;
        case 7: // height
          if (wire_type == VARINT) {
            height_ = static_cast<uint32_t>(proto_buffer.decodeVarint());
            has_height_ = true;
          }
          break;
        default:
          // Skip unknown field
          skipField(proto_buffer, wire_type);
          break;
      }
    }
  }

private:
  void skipField(ProtoBuffer& buffer, WireType wire_type) {
    switch (wire_type) {
      case VARINT:
        buffer.decodeVarint();
        break;
      case FIXED64:
        buffer.decodeFixed64();
        break;
      case LENGTH_DELIMITED:
        buffer.decodeLengthDelimited();
        break;
      case FIXED32:
        buffer.decodeFixed32();
        break;
    }
  }
};
} // namespace sdproto

#include <cmath> // For std::isnan
#include <cstdint>
#include <cstring> // For memcpy
#include <iomanip>
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
  // 比较两个 PointXYZIT 对象是否相等（需要特殊处理 NaN）
  // bool operator==(const PointXYZIT& other) const { return ((std::isnan(x) && std::isnan(other.x)) || x == other.x) && ((std::isnan(y) && std::isnan(other.y)) || y == other.y) && ((std::isnan(z) && std::isnan(other.z)) || z == other.z) && intensity == other.intensity && timestamp == other.timestamp; }
};

// PointCloud 中没有指定 default 值的 optional 字段，使用 std::optional 很合适
struct PointCloud {
  std::optional<std::string> frame_id;
  std::optional<bool> is_dense;
  std::vector<PointXYZIT> point;
  std::optional<double> measurement_time;
  std::optional<uint32_t> width;
  std::optional<uint32_t> height;
  // bool operator==(const PointCloud& other) const { return frame_id == other.frame_id && is_dense == other.is_dense && point == other.point && measurement_time == other.measurement_time && width == other.width && height == other.height; }
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
