#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace shmallocator {

struct Header {
  uint32_t bitseq;
  uint32_t id;
  uint16_t refcount;
  uint32_t size;
  int32_t prev;
  int32_t next;
  bool has_mutex;
  bool is_free;
  pthread_mutex_t mutex;
  pthread_mutexattr_t attr;
};

constexpr uint32_t BITSEQ{12344321};
extern void *shmptr;
extern size_t shmsize;
bool initshm(const std::string &shm_file_path, size_t shm_base_address, size_t shm_max_size);
void *shmalloc(uint32_t size, int *id, const char *filename, int linenumber);
void *shmget(int id, const char *filename, int linenumber);
void shmfree(void *ptr, const char *filename, int linenumber);
int shmgetmaxid();
bool uninitshm();

template <typename T> class Allocator {
#define HAS_MEMBER(member) \
  template <typename SubT> class Has_##member { \
  private: \
    template <typename Class> constexpr static bool Test(decltype(Class::member) *) { \
      return true; \
    } \
    template <typename Class> constexpr static bool Test(...) { \
      return false; \
    } \
\
  public: \
    constexpr static bool value = Test<SubT>(nullptr); \
  }

private:
  HAS_MEMBER(id);

public:
  typedef T value_type;
  T *allocate(size_t size = 1) {
    // printf("allocate %s size %lu\n", typeid(T).name(), size);
    int id{-1};
    T *ptr = (T *)shmalloc(sizeof(T) * size, &id, __FILE__, __LINE__);
    if (!ptr) {
      throw std::bad_alloc();
    }
    for (int i = 0; i < size; ++i) {
      new (ptr + i) T{};
    }
    if constexpr (Has_id<T>::value) {
      for (int i = 0; i < size; ++i) {
        ptr[i].id = id;
      }
    }
    return ptr;
  }
  void deallocate(T *ptr, size_t size = 1) {
    // printf("deallocate %s size %lu\n", typeid(T).name(), size);
    for (int i = 0; i < size; ++i) {
      (ptr + i)->~T();
    }
    shmfree(ptr, __FILE__, __LINE__);
  }
#undef HAS_MEMBER
};
template <typename T1, typename T2> bool operator==(const Allocator<T1> &lhs, const Allocator<T2> &rhs) noexcept {
  return true;
}
template <typename T1, typename T2> bool operator!=(const Allocator<T1> &lhs, const Allocator<T2> &rhs) noexcept {
  return false;
}

template <typename T> T *shmgetobjbytag(const char *tag) {
  int max_id = shmgetmaxid();
  T *pobj{nullptr};
  for (int i = 0; i < max_id; ++i) {
    T *ptr = (T *)shmget(i, __FILE__, __LINE__);
    if (ptr != nullptr && strncmp(ptr->tag, tag, sizeof(T::tag)) == 0) {
      pobj = ptr;
      break;
    }
  }
  if (pobj == nullptr) {
    pobj = Allocator<T>{}.allocate();
    strncpy(pobj->tag, tag, sizeof(T::tag));
  }
  return pobj;
}

template <typename _CharT, typename _Traits>
class shmbasic_string : public std::basic_string<_CharT, _Traits, Allocator<_CharT>> {
public:
  /// aloocator typedef
  typedef Allocator<_CharT> AllocatorType_t;
  /// parent typedef
  typedef std::basic_string<_CharT, _Traits, AllocatorType_t> __parent;
  /// type of size
  typedef typename __parent::size_type size_type;

  int32_t id{-1};

  /**
   * @short Default constructor creates an empty string.
   */
  shmbasic_string() : __parent() {}

  /**
   * @short Construct %shstring as copy of a std::string.
   * @param __str Source string.
   */
  shmbasic_string(const std::basic_string<_CharT, _Traits> &__str)
    : __parent(__str.data(), __str.size(), AllocatorType_t()) {}

  /**
   * @short Construct %shstring as copy of a substring.
   * @param __str Source string.
   * @param __pos Index of first character to copy from.
   * @param __n Number of characters to copy.
   */
  shmbasic_string(const std::basic_string<_CharT, _Traits> &__str, size_type __pos, size_type __n)
    : __parent(__str.data(), __pos, __n, AllocatorType_t()) {}

  /**
   * @short Construct %shstring as copy of a C string.
   * @param __s Source C string.
   */
  shmbasic_string(const _CharT *__s) : __parent(__s, AllocatorType_t()) {}

  /**
   * @short Construct %shstring as multiple characters.
   * @param __n Number of characters.
   * @param __c Character to use.
   */
  shmbasic_string(size_type __n, _CharT __c) : __parent(__n, __c, AllocatorType_t()) {}

  /**
   * @short Construct %shstring as copy of a range.
   * @param __beg Start of range.
   * @param __end End of range.
   */
  template <class _InputIterator>
  shmbasic_string(_InputIterator __beg, _InputIterator __end) : __parent(__beg, __end, AllocatorType_t()) {}
};

/**
 * @short string definition.
 */
typedef shmbasic_string<char, std::char_traits<char>> shmstring;

/**
 * @short wstring definition.
 */
typedef shmbasic_string<wchar_t, std::char_traits<wchar_t>> shmwstring;

template <typename _Tp> class shmvector : public std::vector<_Tp, Allocator<_Tp>> {
public:
  /// aloocator typedef
  typedef Allocator<_Tp> AllocatorType_t;
  /// parent typedef
  typedef std::vector<_Tp, AllocatorType_t> __parent;
  /// type of size
  typedef typename __parent::size_type size_type;
  /// type of size
  typedef typename __parent::value_type value_type;

  int32_t id{-1};

  /**
   * @short Default constructor creates no elements.
   */
  shmvector() : __parent(AllocatorType_t()) {}

  /**
   * @short Create a %shmvector with copies of an exemplar element.
   * @param __n The number of elements to initially create.
   * @param __value An element to copy.
   */
  shmvector(size_type __n, const value_type &__value = value_type()) : __parent(__n, __value, AllocatorType_t()) {}

  /**
   * @short Construct %shmvector from std vector.
   * @param __other other %shmvector.
   */
  template <typename _otherTp, typename _otherAllocT>
  shmvector(const std::vector<_otherTp, _otherAllocT> &__other)
    : __parent(__other.begin(), __other.end(), AllocatorType_t()) {}

  /**
   * @short Builds a %shmvector from a range.
   * @param __first An input iterator.
   * @param __last An input iterator.
   */
  template <typename _InputIterator>
  shmvector(_InputIterator __first, _InputIterator __last) : __parent(__first, __last, AllocatorType_t()) {}
};

class shmmutex {
public:
  int32_t id;
  shmmutex() {
    pthread_mutexattr_init(&m_mutex_attr);
    pthread_mutexattr_settype(&m_mutex_attr, PTHREAD_MUTEX_NORMAL);
    pthread_mutexattr_setpshared(&m_mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutexattr_setrobust(&m_mutex_attr, PTHREAD_MUTEX_ROBUST);
    pthread_mutex_init(&m_mutex, &m_mutex_attr);
  }
  ~shmmutex() {
    pthread_mutexattr_destroy(&m_mutex_attr);
    pthread_mutex_destroy(&m_mutex);
  }
  int lock() {
    int ret = pthread_mutex_lock(&m_mutex);
    if (ret == EOWNERDEAD) {
      printf("EOWNERDEAD\n");
      ret = pthread_mutex_consistent(&m_mutex);
      ret = unlock();
      ret = pthread_mutex_lock(&m_mutex);
    }
    return ret;
  }
  int try_lock() {
    int ret = pthread_mutex_trylock(&m_mutex);
    if (ret == EOWNERDEAD) {
      printf("EOWNERDEAD\n");
      ret = pthread_mutex_consistent(&m_mutex);
      ret = unlock();
      ret = pthread_mutex_trylock(&m_mutex);
    }
    return ret;
  }
  int unlock() {
    return pthread_mutex_unlock(&m_mutex);
  }
  void reset() {
    pthread_mutex_destroy(&m_mutex);
    pthread_mutex_init(&m_mutex, &m_mutex_attr);
  }

private:
  pthread_mutexattr_t m_mutex_attr;
  pthread_mutex_t m_mutex;
  friend class shmcond;
};

class shmcond {
public:
  int32_t id;
  shmcond() {
    pthread_condattr_init(&m_cond_attr);
    pthread_condattr_setpshared(&m_cond_attr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&m_cond, &m_cond_attr);
  }
  ~shmcond() {
    pthread_cond_destroy(&m_cond);
  }
  int wait(shmmutex &m) {
    return pthread_cond_wait(&m_cond, &m.m_mutex);
  }
  int timedwait(const struct timespec &ts, shmmutex &m) {
    return pthread_cond_timedwait(&m_cond, &m.m_mutex, &ts);
  }
  int signal() {
    return pthread_cond_signal(&m_cond);
  }
  int broadcast() {
    return pthread_cond_broadcast(&m_cond);
  }

private:
  pthread_condattr_t m_cond_attr;
  pthread_cond_t m_cond;
};

} // namespace shmallocator
