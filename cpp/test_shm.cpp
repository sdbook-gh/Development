#include <iostream>
#include <atomic>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <new>
#include <memory>
#include <algorithm>
#include <stdexcept>

// 共享内存管理器
class SharedMemoryPool {
public:
  struct Header {
    std::atomic<size_t> offset;
    size_t total_size;
  };

  Header* header_;
  char* data_;
  size_t data_size_;

  SharedMemoryPool(void* base_addr, size_t total_size, bool init = false) {
    header_ = static_cast<Header*>(base_addr);
    data_ = reinterpret_cast<char*>(header_ + 1);
    data_size_ = total_size - sizeof(Header);

    if (init) {
      header_->offset = 0;
      header_->total_size = data_size_;
    }
  }

  void* allocate(size_t size, size_t alignment = 8) {
    // 计算对齐后的偏移
    size_t current_offset = header_->offset.load(std::memory_order_relaxed);
    size_t aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1);
    size_t new_offset = aligned_offset + size;

    // 原子更新偏移
    size_t expected = current_offset;
    while (!header_->offset.compare_exchange_weak(expected, new_offset, std::memory_order_release, std::memory_order_relaxed)) {
      aligned_offset = (expected + alignment - 1) & ~(alignment - 1);
      new_offset = aligned_offset + size;

      if (new_offset > header_->total_size) {
        return nullptr; // 内存不足
      }
    }

    if (new_offset > header_->total_size) {
      // 回滚
      header_->offset.store(expected, std::memory_order_release);
      return nullptr;
    }

    return data_ + aligned_offset;
  }

  void reset() { header_->offset = 0; }
};

// 共享内存中的vector实现
template <typename T>
class ShmVector {
public:
  struct Metadata {
    T* data;
    size_t size;
    size_t capacity;
    SharedMemoryPool* pool;
  };

  Metadata* meta_;

  void grow(size_t new_capacity) {
    if (new_capacity <= meta_->capacity) return;

    // 分配新内存
    T* new_data = static_cast<T*>(meta_->pool->allocate(new_capacity * sizeof(T), alignof(T)));
    if (!new_data) { throw std::bad_alloc(); }

    // 复制数据
    if (meta_->data && meta_->size > 0) {
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(new_data, meta_->data, meta_->size * sizeof(T));
      } else {
        for (size_t i = 0; i < meta_->size; ++i) {
          new (new_data + i) T(std::move(meta_->data[i]));
          meta_->data[i].~T();
        }
      }
    }

    meta_->data = new_data;
    meta_->capacity = new_capacity;
  }

  // 在共享内存中创建vector元数据
  static ShmVector* create(SharedMemoryPool* pool, size_t initial_capacity = 0) {
    void* meta_mem = pool->allocate(sizeof(Metadata), alignof(Metadata));
    if (!meta_mem) { throw std::bad_alloc(); }

    Metadata* meta = new (meta_mem) Metadata{nullptr, 0, 0, pool};

    if (initial_capacity > 0) {
      meta->data = static_cast<T*>(pool->allocate(initial_capacity * sizeof(T), alignof(T)));
      if (!meta->data) { throw std::bad_alloc(); }
      meta->capacity = initial_capacity;
    }

    ShmVector* vec = static_cast<ShmVector*>(pool->allocate(sizeof(ShmVector), alignof(ShmVector)));
    if (!vec) { throw std::bad_alloc(); }

    vec->meta_ = meta;
    return vec;
  }

  // 从已存在的元数据构造（用于跨进程访问）
  explicit ShmVector(Metadata* meta) : meta_(meta) {}

  // 基本操作
  void push_back(const T& value) {
    if (meta_->size >= meta_->capacity) {
      size_t new_cap = meta_->capacity == 0 ? 1 : meta_->capacity * 2;
      grow(new_cap);
    }

    if constexpr (std::is_trivially_copyable_v<T>) {
      meta_->data[meta_->size] = value;
    } else {
      new (meta_->data + meta_->size) T(value);
    }
    meta_->size++;
  }

  void push_back(T&& value) {
    if (meta_->size >= meta_->capacity) {
      size_t new_cap = meta_->capacity == 0 ? 1 : meta_->capacity * 2;
      grow(new_cap);
    }

    if constexpr (std::is_trivially_copyable_v<T>) {
      meta_->data[meta_->size] = std::move(value);
    } else {
      new (meta_->data + meta_->size) T(std::move(value));
    }
    meta_->size++;
  }

  void pop_back() {
    if (meta_->size > 0) {
      meta_->size--;
      if constexpr (!std::is_trivially_destructible_v<T>) { meta_->data[meta_->size].~T(); }
    }
  }

  T& operator[](size_t index) { return meta_->data[index]; }

  const T& operator[](size_t index) const { return meta_->data[index]; }

  T& at(size_t index) {
    if (index >= meta_->size) { throw std::out_of_range("ShmVector: index out of range"); }
    return meta_->data[index];
  }

  const T& at(size_t index) const {
    if (index >= meta_->size) { throw std::out_of_range("ShmVector: index out of range"); }
    return meta_->data[index];
  }

  T* data() { return meta_->data; }
  const T* data() const { return meta_->data; }

  size_t size() const { return meta_->size; }
  size_t capacity() const { return meta_->capacity; }
  bool empty() const { return meta_->size == 0; }

  void clear() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_t i = 0; i < meta_->size; ++i) { meta_->data[i].~T(); }
    }
    meta_->size = 0;
  }

  void reserve(size_t new_capacity) {
    if (new_capacity > meta_->capacity) { grow(new_capacity); }
  }

  void resize(size_t new_size) {
    if (new_size < meta_->size) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        for (size_t i = new_size; i < meta_->size; ++i) { meta_->data[i].~T(); }
      }
    } else if (new_size > meta_->size) {
      if (new_size > meta_->capacity) { grow(new_size); }
      if constexpr (!std::is_trivially_constructible_v<T>) {
        for (size_t i = meta_->size; i < new_size; ++i) { new (meta_->data + i) T(); }
      }
    }
    meta_->size = new_size;
  }

  // 迭代器
  T* begin() { return meta_->data; }
  T* end() { return meta_->data + meta_->size; }
  const T* begin() const { return meta_->data; }
  const T* end() const { return meta_->data + meta_->size; }

  // 获取元数据指针（用于序列化）
  Metadata* get_metadata() const { return meta_; }
};

// SPSC队列，存储ShmVector的元数据指针
template <typename T>
class ShmSPSCQueue {
public:
  using VectorType = ShmVector<T>;
  using MetadataPtr = typename VectorType::Metadata*;

  struct Node {
    std::atomic<bool> ready{false};
    alignas(64) MetadataPtr data;
  };

  struct QueueHeader {
    alignas(64) std::atomic<size_t> write_index;
    alignas(64) std::atomic<size_t> read_index;
    size_t capacity;
    SharedMemoryPool* pool;
    Node nodes[1]; // 柔性数组
  };

  QueueHeader* header_;

  static ShmSPSCQueue* create(SharedMemoryPool* pool, size_t capacity) {
    size_t header_size = sizeof(QueueHeader) + (capacity - 1) * sizeof(Node);
    void* mem = pool->allocate(header_size, 64);
    if (!mem) { throw std::bad_alloc(); }

    QueueHeader* header = new (mem) QueueHeader{};
    header->write_index = 0;
    header->read_index = 0;
    header->capacity = capacity;
    header->pool = pool;

    // 初始化节点
    for (size_t i = 0; i < capacity; ++i) { new (&header->nodes[i]) Node{}; }

    void* queue_mem = pool->allocate(sizeof(ShmSPSCQueue), alignof(ShmSPSCQueue));
    if (!queue_mem) { throw std::bad_alloc(); }

    ShmSPSCQueue* queue = new (queue_mem) ShmSPSCQueue{header};
    queue->header_ = header;
    return queue;
  }

  explicit ShmSPSCQueue(QueueHeader* header) : header_(header) {}

  bool push(const VectorType& vec) {
    size_t current_write = header_->write_index.load(std::memory_order_relaxed);
    size_t next_write = (current_write + 1) % header_->capacity;

    if (next_write == header_->read_index.load(std::memory_order_acquire)) {
      return false; // 队列满
    }

    // 创建vector的副本
    auto* new_vec = VectorType::create(header_->pool, vec.size());
    for (size_t i = 0; i < vec.size(); ++i) { new_vec->push_back(vec[i]); }

    header_->nodes[current_write].data = new_vec->get_metadata();
    header_->nodes[current_write].ready.store(true, std::memory_order_release);
    header_->write_index.store(next_write, std::memory_order_release);
    return true;
  }

  bool pop(VectorType& vec) {
    size_t current_read = header_->read_index.load(std::memory_order_relaxed);

    if (current_read == header_->write_index.load(std::memory_order_acquire)) {
      return false; // 队列空
    }

    while (!header_->nodes[current_read].ready.load(std::memory_order_acquire)) {
      // 等待数据准备好
    }

    vec = VectorType(header_->nodes[current_read].data);
    header_->nodes[current_read].ready.store(false, std::memory_order_release);

    size_t next_read = (current_read + 1) % header_->capacity;
    header_->read_index.store(next_read, std::memory_order_release);
    return true;
  }

  bool empty() const { return header_->read_index.load(std::memory_order_acquire) == header_->write_index.load(std::memory_order_acquire); }
};

// 共享内存管理
class SharedMemoryManager {
private:
  void* shm_addr_;
  size_t shm_size_;
  int shm_fd_;
  std::string shm_name_;
  bool is_creator_;

public:
  SharedMemoryManager(const std::string& name, size_t size, bool create = true, void* addr = nullptr) : shm_name_(name), shm_size_(size), is_creator_(create) {
    if (create) {
      // 删除已存在的共享内存
      shm_unlink(shm_name_.c_str());

      shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
      if (shm_fd_ == -1) { throw std::runtime_error("Failed to create shared memory"); }

      if (ftruncate(shm_fd_, shm_size_) == -1) {
        close(shm_fd_);
        throw std::runtime_error("Failed to set shared memory size");
      }
    } else {
      shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
      if (shm_fd_ == -1) { throw std::runtime_error("Failed to open shared memory"); }
    }

    shm_addr_ = mmap(addr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_addr_ == MAP_FAILED) {
      close(shm_fd_);
      throw std::runtime_error("Failed to map shared memory");
    }
  }

  ~SharedMemoryManager() {
    if (shm_addr_ != nullptr) { munmap(shm_addr_, shm_size_); }
    if (shm_fd_ != -1) { close(shm_fd_); }
    if (is_creator_) { shm_unlink(shm_name_.c_str()); }
  }

  void* get_memory() const { return shm_addr_; }
  size_t get_size() const { return shm_size_; }
};

// 使用示例
int main(int argc, char* argv[]) {
  const size_t SHM_SIZE = 1024 * 1024 * 10; // 10MB
  const size_t QUEUE_SIZE = 100;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " [producer|consumer]" << std::endl;
    return 1;
  }

  std::string mode = argv[1];

  try {
    if (mode == "producer") {
      std::cout << "Starting producer..." << std::endl;

      // 创建共享内存
      SharedMemoryManager shm_manager_reg("/my_reg", 1024, true, (void*)0x3f00000000);
      uint64_t* reg = reinterpret_cast<uint64_t*>(shm_manager_reg.get_memory());
      printf("reg: %p\n", reg);
      memset(reg, 0, sizeof(shm_manager_reg.get_size()));
      SharedMemoryManager shm_manager("/my_spsc_queue", SHM_SIZE, true, (void*)0x3f10000000);
      printf("shm_manager: %p\n", shm_manager.get_memory());
      SharedMemoryPool pool(shm_manager.get_memory(), shm_manager.get_size(), true);

      // 创建队列
      auto* queue = ShmSPSCQueue<int>::create(&pool, QUEUE_SIZE);
      // 保存队列指针
      reg[0] = reinterpret_cast<uint64_t>(queue);
      std::cout << "queue: " << (void*)queue << std::endl;
      std::cout << "queue header: " << queue->header_ << std::endl;
      std::cout << "queue header capacity: " << queue->header_->capacity << std::endl;
      std::cout << "queue header read_index: " << queue->header_->read_index << std::endl;
      std::cout << "queue header write_index: " << queue->header_->write_index << std::endl;
      std::cout << "queue header nodes: " << queue->header_->nodes << std::endl;
      std::cout << "queue header nodes size: " << sizeof(queue->header_->nodes) << std::endl;

      // 生产数据
      for (int i = 0; i < 10; ++i) {
        auto* vec = ShmVector<int>::create(&pool);
        for (int j = 0; j <= i; ++j) { vec->push_back(i * 10 + j); }

        if (queue->push(*vec)) {
          std::cout << "Pushed vector of size " << vec->size() << ": ";
          for (int val : *vec) { std::cout << val << " "; }
          std::cout << std::endl;
        } else {
          std::cout << "Queue is full!" << std::endl;
        }

        usleep(500000); // 0.5秒
      }

      std::cout << "Producer finished. Press Enter to exit..." << std::endl;
      std::cin.get();

    } else if (mode == "consumer") {
      std::cout << "Starting consumer..." << std::endl;

      // 连接到共享内存
      SharedMemoryManager shm_manager_reg("/my_reg", 1024, false, (void*)0x3f00000000);
      uint64_t* reg = reinterpret_cast<uint64_t*>(shm_manager_reg.get_memory());
      printf("reg: %p\n", reg);
      SharedMemoryManager shm_manager("/my_spsc_queue", SHM_SIZE, false, (void*)0x3f10000000);
      printf("shm_manager: %p\n", shm_manager.get_memory());
      SharedMemoryPool pool(shm_manager.get_memory(), shm_manager.get_size(), false);

      // 获取队列指针
      ShmSPSCQueue<int>* queue = reinterpret_cast<ShmSPSCQueue<int>*>(reg[0]);
      std::cout << "queue: " << (void*)queue << std::endl;
      std::cout << "queue header: " << queue->header_ << std::endl;
      std::cout << "queue header capacity: " << queue->header_->capacity << std::endl;
      std::cout << "queue header read_index: " << queue->header_->read_index << std::endl;
      std::cout << "queue header write_index: " << queue->header_->write_index << std::endl;
      std::cout << "queue header nodes: " << queue->header_->nodes << std::endl;
      std::cout << "queue header nodes size: " << sizeof(queue->header_->nodes) << std::endl;

      // 消费数据
      while (true) {
        ShmVector<int> vec(nullptr);
        if (queue->pop(vec)) {
          std::cout << "Popped vector of size " << vec.size() << ": ";
          for (int val : vec) { std::cout << val << " "; }
          std::cout << std::endl;
        } else {
          usleep(100000); // 0.1秒
        }
      }

    } else {
      std::cout << "Invalid mode. Use 'producer' or 'consumer'" << std::endl;
      return 1;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
