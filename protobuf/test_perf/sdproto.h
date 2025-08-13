// https://claude.ai/chat/3f94f4c4-f43f-4adf-92fb-2b788ceb5b20
#pragma once
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
      uint8_t byte = (*buffer_)[read_pos_++];
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
