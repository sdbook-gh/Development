// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: receiver.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_receiver_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_receiver_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3014000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3014000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_receiver_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_receiver_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_receiver_2eproto;
namespace apollo {
class ControlMsg;
class ControlMsgDefaultTypeInternal;
extern ControlMsgDefaultTypeInternal _ControlMsg_default_instance_;
class ReceiverConfig;
class ReceiverConfigDefaultTypeInternal;
extern ReceiverConfigDefaultTypeInternal _ReceiverConfig_default_instance_;
}  // namespace apollo
PROTOBUF_NAMESPACE_OPEN
template<> ::apollo::ControlMsg* Arena::CreateMaybeMessage<::apollo::ControlMsg>(Arena*);
template<> ::apollo::ReceiverConfig* Arena::CreateMaybeMessage<::apollo::ReceiverConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace apollo {

// ===================================================================

class ControlMsg PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:apollo.ControlMsg) */ {
 public:
  inline ControlMsg() : ControlMsg(nullptr) {}
  virtual ~ControlMsg();

  ControlMsg(const ControlMsg& from);
  ControlMsg(ControlMsg&& from) noexcept
    : ControlMsg() {
    *this = ::std::move(from);
  }

  inline ControlMsg& operator=(const ControlMsg& from) {
    CopyFrom(from);
    return *this;
  }
  inline ControlMsg& operator=(ControlMsg&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const ControlMsg& default_instance();

  static inline const ControlMsg* internal_default_instance() {
    return reinterpret_cast<const ControlMsg*>(
               &_ControlMsg_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ControlMsg& a, ControlMsg& b) {
    a.Swap(&b);
  }
  inline void Swap(ControlMsg* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ControlMsg* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ControlMsg* New() const final {
    return CreateMaybeMessage<ControlMsg>(nullptr);
  }

  ControlMsg* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ControlMsg>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ControlMsg& from);
  void MergeFrom(const ControlMsg& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ControlMsg* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "apollo.ControlMsg";
  }
  protected:
  explicit ControlMsg(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_receiver_2eproto);
    return ::descriptor_table_receiver_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kSpeedFieldNumber = 5,
  };
  // optional uint64 speed = 5;
  bool has_speed() const;
  private:
  bool _internal_has_speed() const;
  public:
  void clear_speed();
  ::PROTOBUF_NAMESPACE_ID::uint64 speed() const;
  void set_speed(::PROTOBUF_NAMESPACE_ID::uint64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint64 _internal_speed() const;
  void _internal_set_speed(::PROTOBUF_NAMESPACE_ID::uint64 value);
  public:

  // @@protoc_insertion_point(class_scope:apollo.ControlMsg)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::uint64 speed_;
  friend struct ::TableStruct_receiver_2eproto;
};
// -------------------------------------------------------------------

class ReceiverConfig PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:apollo.ReceiverConfig) */ {
 public:
  inline ReceiverConfig() : ReceiverConfig(nullptr) {}
  virtual ~ReceiverConfig();

  ReceiverConfig(const ReceiverConfig& from);
  ReceiverConfig(ReceiverConfig&& from) noexcept
    : ReceiverConfig() {
    *this = ::std::move(from);
  }

  inline ReceiverConfig& operator=(const ReceiverConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline ReceiverConfig& operator=(ReceiverConfig&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const ReceiverConfig& default_instance();

  static inline const ReceiverConfig* internal_default_instance() {
    return reinterpret_cast<const ReceiverConfig*>(
               &_ReceiverConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(ReceiverConfig& a, ReceiverConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(ReceiverConfig* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ReceiverConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ReceiverConfig* New() const final {
    return CreateMaybeMessage<ReceiverConfig>(nullptr);
  }

  ReceiverConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ReceiverConfig>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ReceiverConfig& from);
  void MergeFrom(const ReceiverConfig& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ReceiverConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "apollo.ReceiverConfig";
  }
  protected:
  explicit ReceiverConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_receiver_2eproto);
    return ::descriptor_table_receiver_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kNameFieldNumber = 1,
    kControlTopicFieldNumber = 2,
  };
  // optional string name = 1;
  bool has_name() const;
  private:
  bool _internal_has_name() const;
  public:
  void clear_name();
  const std::string& name() const;
  void set_name(const std::string& value);
  void set_name(std::string&& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  std::string* mutable_name();
  std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // optional string control_topic = 2;
  bool has_control_topic() const;
  private:
  bool _internal_has_control_topic() const;
  public:
  void clear_control_topic();
  const std::string& control_topic() const;
  void set_control_topic(const std::string& value);
  void set_control_topic(std::string&& value);
  void set_control_topic(const char* value);
  void set_control_topic(const char* value, size_t size);
  std::string* mutable_control_topic();
  std::string* release_control_topic();
  void set_allocated_control_topic(std::string* control_topic);
  private:
  const std::string& _internal_control_topic() const;
  void _internal_set_control_topic(const std::string& value);
  std::string* _internal_mutable_control_topic();
  public:

  // @@protoc_insertion_point(class_scope:apollo.ReceiverConfig)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr control_topic_;
  friend struct ::TableStruct_receiver_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ControlMsg

// optional uint64 speed = 5;
inline bool ControlMsg::_internal_has_speed() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool ControlMsg::has_speed() const {
  return _internal_has_speed();
}
inline void ControlMsg::clear_speed() {
  speed_ = PROTOBUF_ULONGLONG(0);
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 ControlMsg::_internal_speed() const {
  return speed_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 ControlMsg::speed() const {
  // @@protoc_insertion_point(field_get:apollo.ControlMsg.speed)
  return _internal_speed();
}
inline void ControlMsg::_internal_set_speed(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  _has_bits_[0] |= 0x00000001u;
  speed_ = value;
}
inline void ControlMsg::set_speed(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  _internal_set_speed(value);
  // @@protoc_insertion_point(field_set:apollo.ControlMsg.speed)
}

// -------------------------------------------------------------------

// ReceiverConfig

// optional string name = 1;
inline bool ReceiverConfig::_internal_has_name() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool ReceiverConfig::has_name() const {
  return _internal_has_name();
}
inline void ReceiverConfig::clear_name() {
  name_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& ReceiverConfig::name() const {
  // @@protoc_insertion_point(field_get:apollo.ReceiverConfig.name)
  return _internal_name();
}
inline void ReceiverConfig::set_name(const std::string& value) {
  _internal_set_name(value);
  // @@protoc_insertion_point(field_set:apollo.ReceiverConfig.name)
}
inline std::string* ReceiverConfig::mutable_name() {
  // @@protoc_insertion_point(field_mutable:apollo.ReceiverConfig.name)
  return _internal_mutable_name();
}
inline const std::string& ReceiverConfig::_internal_name() const {
  return name_.Get();
}
inline void ReceiverConfig::_internal_set_name(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline void ReceiverConfig::set_name(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:apollo.ReceiverConfig.name)
}
inline void ReceiverConfig::set_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:apollo.ReceiverConfig.name)
}
inline void ReceiverConfig::set_name(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000001u;
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:apollo.ReceiverConfig.name)
}
inline std::string* ReceiverConfig::_internal_mutable_name() {
  _has_bits_[0] |= 0x00000001u;
  return name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* ReceiverConfig::release_name() {
  // @@protoc_insertion_point(field_release:apollo.ReceiverConfig.name)
  if (!_internal_has_name()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return name_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void ReceiverConfig::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:apollo.ReceiverConfig.name)
}

// optional string control_topic = 2;
inline bool ReceiverConfig::_internal_has_control_topic() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool ReceiverConfig::has_control_topic() const {
  return _internal_has_control_topic();
}
inline void ReceiverConfig::clear_control_topic() {
  control_topic_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000002u;
}
inline const std::string& ReceiverConfig::control_topic() const {
  // @@protoc_insertion_point(field_get:apollo.ReceiverConfig.control_topic)
  return _internal_control_topic();
}
inline void ReceiverConfig::set_control_topic(const std::string& value) {
  _internal_set_control_topic(value);
  // @@protoc_insertion_point(field_set:apollo.ReceiverConfig.control_topic)
}
inline std::string* ReceiverConfig::mutable_control_topic() {
  // @@protoc_insertion_point(field_mutable:apollo.ReceiverConfig.control_topic)
  return _internal_mutable_control_topic();
}
inline const std::string& ReceiverConfig::_internal_control_topic() const {
  return control_topic_.Get();
}
inline void ReceiverConfig::_internal_set_control_topic(const std::string& value) {
  _has_bits_[0] |= 0x00000002u;
  control_topic_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline void ReceiverConfig::set_control_topic(std::string&& value) {
  _has_bits_[0] |= 0x00000002u;
  control_topic_.Set(
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::move(value), GetArena());
  // @@protoc_insertion_point(field_set_rvalue:apollo.ReceiverConfig.control_topic)
}
inline void ReceiverConfig::set_control_topic(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000002u;
  control_topic_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(value), GetArena());
  // @@protoc_insertion_point(field_set_char:apollo.ReceiverConfig.control_topic)
}
inline void ReceiverConfig::set_control_topic(const char* value,
    size_t size) {
  _has_bits_[0] |= 0x00000002u;
  control_topic_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, ::std::string(
      reinterpret_cast<const char*>(value), size), GetArena());
  // @@protoc_insertion_point(field_set_pointer:apollo.ReceiverConfig.control_topic)
}
inline std::string* ReceiverConfig::_internal_mutable_control_topic() {
  _has_bits_[0] |= 0x00000002u;
  return control_topic_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* ReceiverConfig::release_control_topic() {
  // @@protoc_insertion_point(field_release:apollo.ReceiverConfig.control_topic)
  if (!_internal_has_control_topic()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000002u;
  return control_topic_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void ReceiverConfig::set_allocated_control_topic(std::string* control_topic) {
  if (control_topic != nullptr) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  control_topic_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), control_topic,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:apollo.ReceiverConfig.control_topic)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace apollo

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_receiver_2eproto