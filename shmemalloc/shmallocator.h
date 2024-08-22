#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace shmallocator {

constexpr uint16_t SUPPORT_VERSION{10};
constexpr uint32_t BITSEQ{12344321};

struct Header {
  uint32_t bitseq; // valid flag
  uint16_t version;
  uint32_t id;
  uint32_t size;
  int32_t prev;
  int32_t next;
  bool has_mutex;
  bool is_free;
  pthread_mutex_t mutex;
};

extern void *shmptr;
extern uint32_t shmsize;
bool initshm(const std::string &shm_file_path, size_t shm_base_address, uint32_t shm_max_size);
void *shmalloc(uint32_t size, int *id);
void *shmget(int id);
void shmfree(void *ptr);
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
  using value_type = T;
  T *allocate_def(bool call_def_contor) {
    // printf("allocate %s size %lu\n", typeid(T).name(), size);
    int id{-1};
    T *ptr = (T *)shmalloc(sizeof(T), &id);
    if (ptr == nullptr) {
      throw std::bad_alloc{};
    }
    if (call_def_contor) {
      new (ptr) T;
    }
    if constexpr (Has_id<T>::value) {
      ptr->id = id;
    }
    return ptr;
  }
  T *allocate_init(const T &init_val) {
    // printf("allocate %s size %lu\n", typeid(T).name(), size);
    int id{-1};
    T *ptr = (T *)shmalloc(sizeof(T), &id);
    if (ptr == nullptr) {
      throw std::bad_alloc{};
    }
    new (ptr) T{init_val};
    if constexpr (Has_id<T>::value) {
      ptr->id = id;
    }
    return ptr;
  }
  T *allocate_array_def(uint32_t size, bool call_def_contor) {
    // printf("allocate %s size %lu\n", typeid(T).name(), size);
    int id{-1};
    T *ptr = (T *)shmalloc(sizeof(T) * size, &id);
    if (!ptr) {
      throw std::bad_alloc();
    }
    for (uint32_t i = 0; call_def_contor && i < size; ++i) {
      new (ptr + i) T;
    }
    if constexpr (Has_id<T>::value) {
      for (size_t i = 0; i < size; ++i) {
        ptr[i].id = id;
      }
    }
    return ptr;
  }
  T *allocate_array_init(uint32_t size, const T &init_val) {
    // printf("allocate %s size %lu\n", typeid(T).name(), size);
    int id{-1};
    T *ptr = (T *)shmalloc(sizeof(T) * size, &id);
    if (!ptr) {
      throw std::bad_alloc();
    }
    for (uint32_t i = 0; i < size; ++i) {
      new (ptr + i) T{init_val};
    }
    if constexpr (Has_id<T>::value) {
      for (uint32_t i = 0; i < size; ++i) {
        ptr[i].id = id;
      }
    }
    return ptr;
  }
  void deallocate_def(T *ptr, bool call_detor) {
    // printf("deallocate %s size %lu\n", typeid(T).name(), size);
    if (call_detor) {
      ptr->~T();
    }
    shmfree(ptr);
  }
  void deallocate_array(T *ptr, uint32_t size) {
    // printf("deallocate %s size %lu\n", typeid(T).name(), size);
    for (uint32_t i = 0; i < size; ++i) {
      (ptr + i)->~T();
    }
    shmfree(ptr);
  }
  T *allocate(std::size_t n) {
    return allocate_array_def(n, true);
  }
  void deallocate(T *p, std::size_t n) {
    deallocate_array(p, n);
  }
#undef HAS_MEMBER
};
template <typename T1, typename T2> bool operator==(const Allocator<T1> &lhs, const Allocator<T2> &rhs) noexcept {
  return true;
}
template <typename T1, typename T2> bool operator!=(const Allocator<T1> &lhs, const Allocator<T2> &rhs) noexcept {
  return false;
}

// namespace slab {
// constexpr uint32_t MAX_CHUNKS{8};
// constexpr uint32_t FREE_THRESHOLD{16};
// struct chunk_t {
//   bool flag{false};
//   void *value{nullptr};
// };
// struct slab_t {
//   chunk_t chunks[MAX_CHUNKS];
//   uint32_t free{MAX_CHUNKS};
//   slab_t *next{nullptr};
// };
// struct slabs_t {
//   slab_t *slab_head{nullptr};
//   slab_t *slab_tail{nullptr};
//   slabs_t *next{nullptr};
//   uint32_t slabs{0};
// };
// struct cache_t {
//   // uint32_t id{0};
//   uint32_t size{0};
//   uint32_t chunks{0};
//   uint32_t slabs{0};
//   slabs_t *slabs_full_head{nullptr};
//   slabs_t *slabs_partial_head{nullptr};
//   slabs_t *slabs_empty_head{nullptr};
//   cache_t *next{nullptr};
// };
// class SlabManager {
// private:
//   cache_t *m_cache_chain{nullptr};
//   cache_t *add_cache(uint32_t size) {
//     cache_t *cc{nullptr};
//     cache_t *cp{nullptr};
//     if (m_cache_chain == nullptr) {
//       m_cache_chain = Allocator<cache_t>{}.allocate();
//       cc = m_cache_chain;
//       cc->size = size;
//       cc->chunks = MAX_CHUNKS;
//       cc->slabs_full_head = Allocator<slabs_t>{}.allocate();
//       cc->slabs_partial_head = Allocator<slabs_t>{}.allocate();
//       cc->slabs_empty_head = Allocator<slabs_t>{}.allocate();
//       return cc;
//     } else {
//       for (cc = m_cache_chain; cc != nullptr; cc = cc->next) {
//         if (cc->size > size) {
//           if (cp == nullptr) {
//             m_cache_chain = Allocator<cache_t>{}.allocate();
//             cp = m_cache_chain;
//             cp->next = cc;
//           } else {
//             cp->next = Allocator<cache_t>{}.allocate();
//             cp->next->next = cc;
//             cp = cp->next;
//           }
//           cp->size = size;
//           cp->chunks = MAX_CHUNKS;
//           cp->slabs_full_head = Allocator<slabs_t>{}.allocate();
//           cp->slabs_partial_head = Allocator<slabs_t>{}.allocate();
//           cp->slabs_empty_head = Allocator<slabs_t>{}.allocate();
//           return cp;
//         }
//         cp = cc;
//       }
//       cp->next = Allocator<cache_t>{}.allocate();
//       cp = cp->next;
//       cp->size = size;
//       cp->chunks = MAX_CHUNKS;
//       cp->slabs_full_head = Allocator<slabs_t>{}.allocate();
//       cp->slabs_partial_head = Allocator<slabs_t>{}.allocate();
//       cp->slabs_empty_head = Allocator<slabs_t>{}.allocate();
//     }
//     return cp;
//   }
//   template <typename T> cache_t *cache_match() {
//     uint32_t size{sizeof(T)};
//     cache_t *cc = m_cache_chain;
//     while (cc != nullptr && cc->size != size) {
//       cc = cc->next;
//     }
//     if (cc == nullptr) {
//       return add_cache(size);
//     }
//     return cc;
//   }
//   slab_t *new_slab(uint32_t size) {
//     slab_t *sl = Allocator<slab_t>{}.allocate();
//     for (auto count = 0u; count < MAX_CHUNKS; count++) {
//       sl->chunks[count].value = Allocator<uint8_t>{}.allocate(size, false);
//     }
//     return sl;
//   }
//   int delete_slab(slab_t *sl) {
//     for (auto count = 0u; count < MAX_CHUNKS; count++) {
//       Allocator<uint8_t>{}.deallocate((uint8_t *)sl->chunks[count].value);
//     }
//     Allocator<slab_t>{}.deallocate(sl);
//     return 0;
//   }
//   int slab_queue_remove(slabs_t *sls, slab_t *sl) {
//     slab_t *tmp_sl;
//     slab_t *tmp_sl2;
//     sls->slabs--;
//     /* only one slab */
//     if (sls->slab_head == sls->slab_tail) {
//       sls->slab_head = nullptr;
//       sls->slab_tail = nullptr;
//       sl->next = nullptr;
//       return 0;
//     }
//     /* sl is head */
//     if (sls->slab_head == sl) {
//       sls->slab_head = sls->slab_head->next;
//       sl->next = nullptr;
//       return 0;
//     }
//     tmp_sl = sls->slab_head;
//     tmp_sl2 = tmp_sl;
//     while (tmp_sl != sl) {
//       tmp_sl2 = tmp_sl;
//       tmp_sl = tmp_sl->next;
//     }
//     /* sl is tail */
//     if (sl == sls->slab_tail) {
//       tmp_sl2->next = nullptr;
//       sl->next = nullptr;
//       return 0;
//     }
//     /* other */
//     tmp_sl2->next = sl->next;
//     sl->next = nullptr;
//     return 0;
//   }
//   int slab_queue_add(slabs_t *sls, slab_t *sl) {
//     sls->slabs++;
//     /* the queue is empty */
//     if (sls->slab_head == nullptr) {
//       sls->slab_head = sl;
//       sls->slab_tail = sl;
//       sl->next = nullptr;
//       return 0;
//     }
//     /* Or, add this slab to tail */
//     sls->slab_tail->next = sl;
//     sls->slab_tail = sl;
//     sl->next = nullptr;
//     return 0;
//   }

// public:
//   template <typename T> T *cache_alloc(cache_t *&cp) {
//     if ((cp = cache_match<T>()) == nullptr) {
//       throw std::bad_alloc{};
//     }
//     if (cp->slabs_empty_head->slab_head != nullptr) {
//       auto *tmp_sl = cp->slabs_partial_head->slab_head;
//       tmp_sl->chunks[0].flag = true;
//       tmp_sl->free--;
//       slab_queue_remove(cp->slabs_empty_head, tmp_sl);
//       slab_queue_add(cp->slabs_partial_head, tmp_sl);
//       return (T *)tmp_sl->chunks[0].value;
//     }
//     if (cp->slabs_partial_head->slab_head == nullptr) {
//       auto *sl = new_slab(cp->size);
//       cp->slabs++;
//       cp->slabs_partial_head->slab_head = sl;
//       cp->slabs_partial_head->slab_tail = sl;
//     }
//     auto *tmp_sl = cp->slabs_partial_head->slab_head;
//     while (tmp_sl != nullptr) {
//       for (auto count = 0u; count < MAX_CHUNKS; count++) {
//         if (tmp_sl->chunks[count].flag == false) {
//           tmp_sl->chunks[count].flag = true;
//           tmp_sl->free--;
//           if (tmp_sl->free == 0) {
//             slab_queue_remove(cp->slabs_partial_head, tmp_sl);
//             slab_queue_add(cp->slabs_full_head, tmp_sl);
//           }
//           return (T *)tmp_sl->chunks[count].value;
//         }
//       }
//       tmp_sl = tmp_sl->next;
//     }
//     throw std::bad_alloc{};
//   }
//   void cache_free(cache_t *cp, void *buf) {
//     slab_t *sl;
//     if (cp->slabs_full_head != nullptr) {
//       sl = cp->slabs_full_head->slab_head;
//       while (sl) {
//         for (auto count = 0u; count < MAX_CHUNKS; count++) {
//           if (buf == sl->chunks[count].value) {
//             sl->chunks[count].flag = false;
//             sl->free++;
//             slab_queue_remove(cp->slabs_full_head, sl);
//             slab_queue_add(cp->slabs_partial_head, sl);
//             return;
//           }
//         }
//         sl = sl->next;
//       }
//     }
//     if (cp->slabs_partial_head != nullptr) {
//       sl = cp->slabs_partial_head->slab_head;
//       while (sl) {
//         for (auto count = 0u; count < MAX_CHUNKS; count++) {
//           if (buf == sl->chunks[count].value) {
//             sl->chunks[count].flag = false;
//             sl->free++;
//             if (sl->free == MAX_CHUNKS) {
//               if (cp->slabs_empty_head->slabs >= FREE_THRESHOLD) {
//                 delete_slab(sl);
//                 cp->slabs--;
//                 return;
//               }
//               slab_queue_remove(cp->slabs_partial_head, sl);
//               slab_queue_add(cp->slabs_empty_head, sl);
//             }
//             return;
//           }
//         }
//         sl = sl->next;
//       }
//     }
//     printf("bad pointer\n");
//   }
// };
// } // namespace slab

template <typename T> T *aligned_alloc(void *buffer, uint32_t buffer_size, std::size_t alignment = alignof(T)) {
  void *ptr = buffer;
  size_t size = buffer_size;
  if (std::align(alignment, sizeof(T), ptr, size) != nullptr) {
    T *result = (T *)(ptr);
    return result;
  }
  return nullptr;
}

using ObjectTag = char[64];
template <typename T> T *shmgetobjbytag(const char *tag, bool create = true) {
  int max_id = shmgetmaxid();
  T *pobj{nullptr};
  for (int i = 2; i <= max_id; ++i) {
    T *ptr = (T *)shmget(i);
    if (ptr != nullptr && strncmp(ptr->tag, tag, sizeof(T::tag)) == 0) {
      pobj = ptr;
      break;
    }
  }
  if (pobj == nullptr && create) {
    pobj = Allocator<T>{}.allocate_def(true);
    strncpy(pobj->tag, tag, sizeof(T::tag) - 1);
  }
  return pobj;
}
template <typename T> T *shmgetobjbytag(const char *tag, const T &init_value) {
  int max_id = shmgetmaxid();
  T *pobj{nullptr};
  for (int i = 0; i < max_id; ++i) {
    T *ptr = (T *)shmget(i);
    if (ptr != nullptr && strncmp(ptr->tag, tag, sizeof(T::tag)) == 0) {
      pobj = ptr;
      break;
    }
  }
  if (pobj == nullptr) {
    pobj = Allocator<T>{}.allocate_init(init_value);
    strncpy(pobj->tag, tag, sizeof(T::tag) - 1);
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
    if ((m_pmutex = aligned_alloc<pthread_mutex_t>(buffer, sizeof(buffer))) != nullptr) {
      pthread_mutexattr_init(&m_mutex_attr);
      pthread_mutexattr_settype(&m_mutex_attr, PTHREAD_MUTEX_NORMAL);
      pthread_mutexattr_setpshared(&m_mutex_attr, PTHREAD_PROCESS_SHARED);
      pthread_mutexattr_setrobust(&m_mutex_attr, PTHREAD_MUTEX_ROBUST);
      pthread_mutex_init(m_pmutex, &m_mutex_attr);
    } else {
      throw std::runtime_error{"aligned_alloc error"};
    }
  }
  ~shmmutex() {
    pthread_mutexattr_destroy(&m_mutex_attr);
    pthread_mutex_destroy(m_pmutex);
  }
  int lock() {
    int ret = pthread_mutex_lock(m_pmutex);
    if (ret == EOWNERDEAD) {
      printf("EOWNERDEAD\n");
      ret = pthread_mutex_consistent(m_pmutex);
      ret = unlock();
      ret = pthread_mutex_lock(m_pmutex);
    }
    return ret;
  }
  int try_lock() {
    int ret = pthread_mutex_trylock(m_pmutex);
    if (ret == EOWNERDEAD) {
      printf("EOWNERDEAD\n");
      ret = pthread_mutex_consistent(m_pmutex);
      ret = unlock();
      ret = pthread_mutex_trylock(m_pmutex);
    }
    return ret;
  }
  int unlock() {
    return pthread_mutex_unlock(m_pmutex);
  }
  void reset() {
    pthread_mutex_destroy(m_pmutex);
    pthread_mutex_init(m_pmutex, &m_mutex_attr);
  }

private:
  pthread_mutexattr_t m_mutex_attr;
  pthread_mutex_t *m_pmutex{nullptr};
  pthread_cond_t buffer[2];
};

class shmsemaphore {
public:
  int32_t id;
  shmsemaphore(uint32_t size = 1) {
    if ((m_psem = aligned_alloc<sem_t>(buffer, sizeof(buffer))) != nullptr) {
      m_size = size;
      sem_init(m_psem, 1, m_size);
    } else {
      throw std::runtime_error{"aligned_alloc error"};
    }
  }
  shmsemaphore(const shmsemaphore &other) {
    if ((m_psem = aligned_alloc<sem_t>(buffer, sizeof(buffer))) != nullptr) {
      m_size = other.m_size;
      sem_init(m_psem, 1, m_size);
    } else {
      throw std::runtime_error{"aligned_alloc error"};
    }
  }
  ~shmsemaphore() {
    sem_destroy(m_psem);
  }
  int wait() {
    return sem_wait(m_psem);
  }
  int timedwait(const timespec &ts) {
    return sem_timedwait(m_psem, &ts);
  }
  int try_wait() {
    return sem_trywait(m_psem);
  }
  int signal() {
    return sem_post(m_psem);
  }
  int get_value() {
    int val{0};
    sem_getvalue(m_psem, &val);
    return get_value();
  }
  void reset(int resource_value = 1) {
    sem_destroy(m_psem);
    sem_init(m_psem, 1, resource_value);
  }

private:
  sem_t *m_psem{nullptr};
  sem_t buffer[2];
  int m_size{1};
};

using shm_spin_mutex = boost::interprocess::interprocess_mutex;
using shm_spin_mutex_lock = boost::interprocess::scoped_lock<shm_spin_mutex>;
using shm_spin_cond = boost::interprocess::interprocess_condition;

class shmcond {
public:
  int32_t id;
  shmcond() {
    m_free_semaphore_list.resize(64);
    for (size_t i = 0; i < m_free_semaphore_list.size(); ++i) {
      m_free_semaphore_list[i] = Allocator<shmsemaphore>{}.allocate_init(0);
    }
    m_wait_semaphore_list.clear();
  }
  ~shmcond() {
    for (auto &element : m_free_semaphore_list) {
      Allocator<shmsemaphore>{}.deallocate_def(element, true);
    }
    for (auto &element : m_wait_semaphore_list) {
      Allocator<shmsemaphore>{}.deallocate_def(element, true);
    }
  }
  int wait(shmmutex &m) {
    shmsemaphore *psemaphore{nullptr};
    {
      std::lock_guard<shmmutex> lock{m_mutex};
      if (m_free_semaphore_list.empty()) {
        printf("too many waiters\n");
        return 1;
      }
      psemaphore = *m_free_semaphore_list.rbegin();
      m_free_semaphore_list.pop_back();
      m_wait_semaphore_list.emplace_back(psemaphore);
    }
    m.unlock();
    psemaphore->wait();
    m.lock();
    return 0;
  }
  int timedwait(const timespec &ts, shmmutex &m) {
    shmsemaphore *psemaphore{nullptr};
    {
      std::lock_guard<shmmutex> lock{m_mutex};
      if (m_free_semaphore_list.empty()) {
        printf("too many waiters\n");
        return 1;
      }
      psemaphore = *m_free_semaphore_list.rbegin();
      m_free_semaphore_list.pop_back();
      m_wait_semaphore_list.emplace_back(psemaphore);
    }
    m.unlock();
    psemaphore->timedwait(ts);
    m.lock();
    return 0;
  }
  int signal() {
    shmsemaphore *psemaphore{nullptr};
    {
      std::lock_guard<shmmutex> lock{m_mutex};
      if (!m_wait_semaphore_list.empty()) {
        psemaphore = m_wait_semaphore_list[0];
        m_wait_semaphore_list[0]->signal();
        m_wait_semaphore_list.erase(m_wait_semaphore_list.begin());
        m_free_semaphore_list.emplace_back(psemaphore);
      }
      return 0;
    }
  }
  int broadcast() {
    shmsemaphore *psemaphore{nullptr};
    {
      std::lock_guard<shmmutex> lock{m_mutex};
      while (!m_wait_semaphore_list.empty()) {
        psemaphore = m_wait_semaphore_list[0];
        m_wait_semaphore_list[0]->signal();
        m_wait_semaphore_list.erase(m_wait_semaphore_list.begin());
        m_free_semaphore_list.emplace_back(psemaphore);
      }
      return 0;
    }
  }

private:
  shmmutex m_mutex;
  shmvector<shmsemaphore *> m_wait_semaphore_list;
  shmvector<shmsemaphore *> m_free_semaphore_list;
};

template <typename T> class shmqueue {
private:
  struct Record {
    Record() {}
    Record(const Record &other) {
      m_sequence.store(other.m_sequence.load(std::memory_order_acquire), std::memory_order_release);
      m_data = other.m_data;
    }
    std::atomic<uint32_t> m_sequence{0};
    T m_data;
  };
  static inline uint32_t nextPowerOfTwo(uint32_t m_buffersize) {
    uint32_t result = m_buffersize - 1;
    for (uint32_t i = 1; i <= sizeof(void *) * 4; i <<= 1) {
      result |= result >> i;
    }
    return result + 1;
  }

public:
  shmqueue(uint32_t buffersize) {
    m_buffer.resize(nextPowerOfTwo(buffersize));
    m_buffer_mask = (nextPowerOfTwo(buffersize) - 1);
    buffersize = m_buffer_mask + 1;
    m_enqueue_pos.store(0, std::memory_order_relaxed);
    m_dequeue_pos.store(0, std::memory_order_relaxed);
    for (uint32_t i = 0; i != buffersize; i += 1) {
      m_buffer[i].m_sequence.store(i, std::memory_order_relaxed);
    }
  }

  ~shmqueue() {}

  uint32_t size() const {
    uint32_t head = m_dequeue_pos.load(std::memory_order_acquire);
    return m_enqueue_pos.load(std::memory_order_relaxed) - head;
  }

  uint32_t capacity() const {
    return m_buffer_mask + 1;
  }

  bool push(T const &data) {
    Record *record;
    uint32_t pos = m_enqueue_pos.load(std::memory_order_relaxed);
    for (;;) {
      record = &m_buffer[pos & m_buffer_mask];
      uint32_t seq = record->m_sequence.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)pos;
      if (dif == 0) {
        if (m_enqueue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          break;
        }
      } else if (dif < 0) {
        // printf("queue size %d\n", size());
        return false;
      } else {
        pos = m_enqueue_pos.load(std::memory_order_relaxed);
      }
    }
    record->m_data = data;
    record->m_sequence.store(pos + 1, std::memory_order_release);
    m_semaphore.signal();
    // printf("queue size %d\n", size());
    return true;
  }

  void pop(T &data) {
    while (true) {
      bool need_wait{false};
      Record *record;
      uint32_t pos = m_dequeue_pos.load(std::memory_order_relaxed);
      for (;;) {
        record = &m_buffer[pos & m_buffer_mask];
        uint32_t seq = record->m_sequence.load(std::memory_order_acquire);
        intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
        if (dif == 0) {
          if (m_dequeue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
            break;
          }
        } else if (dif < 0) {
          need_wait = true;
          break;
        } else {
          pos = m_dequeue_pos.load(std::memory_order_relaxed);
        }
      }
      if (!need_wait) {
        data = record->m_data;
        record->m_sequence.store(pos + m_buffer_mask + 1, std::memory_order_release);
        return;
      }
      m_semaphore.wait();
    }
  }

private:
  shmvector<Record> m_buffer;
  uint32_t m_buffer_mask;
  std::atomic<uint32_t> m_enqueue_pos;
  std::atomic<uint32_t> m_dequeue_pos;
  shmqueue(shmqueue const &);
  void operator=(shmqueue const &);
  shmsemaphore m_semaphore;
};

class AliveCheck {
public:
  AliveCheck() {
    if ((m_pmutex = aligned_alloc<pthread_mutex_t>(buffer, sizeof(buffer))) != nullptr) {
      pthread_mutexattr_init(&m_mutex_attr);
      pthread_mutexattr_settype(&m_mutex_attr, PTHREAD_MUTEX_RECURSIVE);
      pthread_mutexattr_setpshared(&m_mutex_attr, PTHREAD_PROCESS_SHARED);
      pthread_mutexattr_setrobust(&m_mutex_attr, PTHREAD_MUTEX_ROBUST);
      pthread_mutex_init(m_pmutex, &m_mutex_attr);
      pthread_mutex_lock(m_pmutex);
    } else {
      throw std::runtime_error{"aligned_alloc error"};
    }
  }
  ~AliveCheck() {
    pthread_mutex_unlock(m_pmutex);
    pthread_mutexattr_destroy(&m_mutex_attr);
    pthread_mutex_destroy(m_pmutex);
  }
  int is_alive() {
    if (status.load() == -1) {
      return -1;
    }
    int ret = pthread_mutex_trylock(m_pmutex);
    // printf("is_alive pthread_mutex_trylock %d\n", ret);
    if (ret == 0) {
      // printf("owner\n");
      status.store(1);
      return 0;
    } else if (ret == EOWNERDEAD) {
      // printf("owner dead\n");
      status.store(-1);
      return -1;
    }
    // printf("owner alive\n");
    return 1;
  }
  int reset() {
    int8_t dead_status{-1};
    if (status.compare_exchange_strong(dead_status, 1)) {
      pthread_mutex_destroy(m_pmutex);
      pthread_mutex_init(m_pmutex, &m_mutex_attr);
      return pthread_mutex_lock(m_pmutex);
    }
    return -1;
  }

private:
  pthread_mutexattr_t m_mutex_attr;
  pthread_mutex_t *m_pmutex{nullptr};
  pthread_cond_t buffer[2];
  std::atomic_int8_t status{0};
};

class AliveMonitor {
public:
  enum Type { PRODUCER, CONSUMER };
  class TrackRecord {
  public:
    enum Operation : int8_t { UNKNOWN = 0, RUNNING, MONITORING };
    TrackRecord() = default;
    TrackRecord(Operation last_operation, const std::string &name) {
      this->last_operation = last_operation;
      strncpy(this->name, name.c_str(), sizeof(this->name) - 1);
    }
    TrackRecord(const TrackRecord &element) {
      this->last_operation.store(element.last_operation);
      strncpy(this->name, element.name, sizeof(this->name) - 1);
    }
    std::atomic_int last_operation{UNKNOWN};
    uint8_t count{0};
    char name[64]{0};
  };

private:
  using TrackRecordVec_t = shmvector<TrackRecord *>;
  shmmutex m_monitor_mutex;
  shmmutex m_producer_vec_mutex;
  TrackRecordVec_t m_producer_vec;
  shmmutex m_consumer_vec_mutex;
  TrackRecordVec_t m_consumer_vec;
  bool m_any_producer_dead{true};
  bool m_any_consumer_dead{true};
  static constexpr int CHECK_INTERVAL_MS{100};
  static constexpr int CHECK_MAX_COUNT{2};
  typedef struct {
    shmallocator::ObjectTag tag{0};
    AliveCheck m_alivecheck;
  } MonitorStatus_t;
  MonitorStatus_t *m_pmonitorstatus{nullptr};

public:
  AliveMonitor() {}
  bool start_monitor() {
    m_pmonitorstatus = shmgetobjbytag<MonitorStatus_t>("monitor_status", true);
    printf("monitor_status %p\n", m_pmonitorstatus);
    if (m_pmonitorstatus->m_alivecheck.is_alive() < 0) {
      m_pmonitorstatus->m_alivecheck.reset();
    }
    if (m_pmonitorstatus->m_alivecheck.is_alive() == 0) {
      printf("monitor running\n");
      auto check_alive = [](TrackRecord *pRecord) {
        if (pRecord->last_operation == TrackRecord::UNKNOWN) {
          pRecord->last_operation = TrackRecord::MONITORING;
          pRecord->count = 0;
          return 0;
        } else if (pRecord->last_operation == TrackRecord::MONITORING) {
          if (pRecord->count >= CHECK_MAX_COUNT) {
            return -1;
          }
          pRecord->count++;
        } else {
          pRecord->last_operation = TrackRecord::MONITORING;
          pRecord->count = 0;
        }
        return 1;
      };
      auto check = [check_alive](shmmutex &mutex, TrackRecordVec_t &vec, bool &dead_flag, Type type) {
        if (mutex.try_lock() == 0) {
          int index{0};
          int res;
          for (int i = 0; i < (int)vec.size(); ++i) {
            if ((res = check_alive(vec[i])) != 1) {
              index = -(i + 1);
              break;
            }
            index++;
          }
          if (index <= 0) {
            dead_flag = true;
            int n = vec.empty() ? -1 : -index - 1;
            if (n >= 0) {
              if (res == 0) {
                printf("%s %d joined\n", vec[n]->name, (int)type);
              } else {
                printf("%s %d dead\n", vec[n]->name, (int)type);
                Allocator<TrackRecord>{}.deallocate_def(*(vec.begin() + n), true);
                vec.erase(vec.begin() + n);
              }
            }
          } else {
            dead_flag = false;
          }
          mutex.unlock();
        }
      };
      while (true) {
        check(m_producer_vec_mutex, m_producer_vec, m_any_producer_dead, PRODUCER);
        check(m_consumer_vec_mutex, m_consumer_vec, m_any_consumer_dead, CONSUMER);
        std::this_thread::sleep_for(std::chrono::milliseconds(CHECK_INTERVAL_MS));
      }
      return true;
    }
    return false;
  }
  bool start_heartbeat(Type type, const std::string &name) {
    m_pmonitorstatus = shmgetobjbytag<MonitorStatus_t>("monitor_status", false);
    printf("monitor_status %p\n", m_pmonitorstatus);
    while (m_pmonitorstatus == nullptr || m_pmonitorstatus->m_alivecheck.is_alive() < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds{CHECK_INTERVAL_MS});
    }
    static bool res = [this, type, name] {
      TrackRecord *pRecord{nullptr};
      AliveMonitor::TrackRecordVec_t *pvec{nullptr};
      shmmutex *pmutex{nullptr};
      if (type == PRODUCER) {
        pvec = &m_producer_vec;
        pmutex = &m_producer_vec_mutex;
      } else {
        pvec = &m_consumer_vec;
        pmutex = &m_consumer_vec_mutex;
      }
      {
        std::lock_guard<shmallocator::shmmutex> lock(*pmutex);
        for (auto it = pvec->begin(); it != pvec->end(); ++it) {
          if (strncmp((*it)->name, name.c_str(), sizeof((*it)->name)) == 0) {
            (*it)->last_operation = TrackRecord::UNKNOWN;
            pRecord = *it;
            break;
          }
        }
      }
      if (pRecord == nullptr) {
        std::lock_guard<shmmutex> lock{*pmutex};
        pvec->emplace_back(shmallocator::Allocator<TrackRecord>{}.allocate_def(true));
        (*pvec->rbegin())->last_operation = TrackRecord::UNKNOWN;
        strncpy((*pvec->rbegin())->name, name.c_str(), sizeof(TrackRecord::name));
        pRecord = *pvec->rbegin();
      }
      std::thread heatbeat_thread([pRecord, this] {
        while (true) {
          if (m_pmonitorstatus->m_alivecheck.is_alive() < 0) {
            printf("monitor is dead, exit\n");
            exit(EXIT_FAILURE);
          }
          int status = TrackRecord::UNKNOWN;
          if (pRecord->last_operation.compare_exchange_strong(status, TrackRecord::UNKNOWN)) {
          } else {
            pRecord->last_operation.store(TrackRecord::RUNNING);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds{CHECK_INTERVAL_MS});
        }
      });
      heatbeat_thread.detach();
      return true;
    }();
    return res;
  }
  // bool any_producer_dead() { return m_any_producer_dead; }
  // bool any_consumer_dead() { return m_any_consumer_dead; }
  // void wait_for_any_producer_alive() {
  //   while (true) {
  //     if (!any_producer_dead()) {
  //       return;
  //     }
  //     std::this_thread::sleep_for(std::chrono::milliseconds{CHECK_INTERVAL_MS});
  //   }
  // }
  // void wait_for_any_consumer_dead() {
  //   while (true) {
  //     if (any_consumer_dead()) {
  //       return;
  //     }
  //     std::this_thread::sleep_for(std::chrono::milliseconds{CHECK_INTERVAL_MS});
  //   }
  // }
};

} // namespace shmallocator
