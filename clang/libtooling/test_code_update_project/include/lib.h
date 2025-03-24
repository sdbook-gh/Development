#pragma once

#include "def.h"
#include "log.h"
#include <map>
#include <memory>

namespace NM {

class NMAnotherClass {
public:
  template <typename T> std::shared_ptr<T> get_instance(const T &in) {
    return std::make_shared<T>(in);
  }
};

class NMClass {
public:
  NMClass();

  enum Enum {
    ONE = 1,
    TWO = 2,
  };
  enum class EnumClass {
    CL_ONE = 1,
    CL_TWO = 2,
  };

  void set_NM_value(int value);
  int get_NM_value() const {
    increment_NM_value();
    return _NM_value;
  }
  std::shared_ptr<open::nmutils::nmlog::NMAsyncLog>
  adjust_NM_log(std::shared_ptr<open::nmutils::nmlog::NMAsyncLog> log,
                std::shared_ptr<NMAnotherClass> another);

private:
  void increment_NM_value() const;
  mutable int _NM_value;
};

} // namespace NM

namespace open {
template <typename T> class NM_managed_ptr {
private:
  T *ptr;             // 指向管理的对象
  std::size_t *count; // 引用计数指针
  // 帮助函数：增加引用计数
  void increment_count() {
    if (count != nullptr) {
      ++(*count);
    }
  }
  // 帮助函数：减少引用计数，如果计数为零则删除对象和计数器
  void decrement_count() {
    if (count != nullptr) {
      --(*count);
      if (*count == 0) {
        delete ptr;
        delete count;
        ptr = nullptr;
        count = nullptr;
      }
    }
  }

public:
  // 默认构造函数：初始化为空指针
  NM_managed_ptr() noexcept : ptr(nullptr), count(nullptr) {}
  // 构造函数：接受一个原始指针
  explicit NM_managed_ptr(T *raw_ptr) noexcept
      : ptr(raw_ptr), count(new std::size_t(1)) {}
  // 拷贝构造函数
  NM_managed_ptr(const NM_managed_ptr &other) noexcept {
    ptr = other.ptr;
    count = other.count;
    increment_count();
  }
  // 移动构造函数
  NM_managed_ptr(NM_managed_ptr &&other) noexcept {
    ptr = other.ptr;
    count = other.count;
    other.ptr = nullptr;
    other.count = nullptr;
  }
  // 析构函数
  ~NM_managed_ptr() { decrement_count(); }
  // 赋值操作符：拷贝赋值
  NM_managed_ptr &operator=(const NM_managed_ptr &other) noexcept {
    if (this != &other) {
      decrement_count();
      ptr = other.ptr;
      count = other.count;
      increment_count();
    }
    return *this;
  }
  // 赋值操作符：移动赋值
  NM_managed_ptr &operator=(NM_managed_ptr &&other) noexcept {
    if (this != &other) {
      decrement_count();
      ptr = other.ptr;
      count = other.count;
      other.ptr = nullptr;
      other.count = nullptr;
    }
    return *this;
  }
  // 解引用操作符
  T &operator*() const noexcept { return *ptr; }
  // 箭头操作符
  T *operator->() const noexcept { return ptr; }
  // 获取原始指针
  T *get() const noexcept { return ptr; }
  // 重置指针
  void reset(T *new_ptr = nullptr) {
    if (ptr != new_ptr) {
      decrement_count();
      ptr = new_ptr;
      count = new std::size_t(1);
    }
  }
  // 获取引用计数
  std::size_t use_count() const noexcept {
    return (count != nullptr) ? *count : 0;
  }
  // 检查是否为空
  bool operator==(std::nullptr_t) const noexcept { return ptr == nullptr; }
  bool operator!=(std::nullptr_t) const noexcept { return ptr != nullptr; }
};

} // namespace open
