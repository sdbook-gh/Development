#include <atomic>
#include <cstdio>
#include <cstring>

class MemoryPoolSPMC {
public:
  void init() {}
  void write(const void *header, uint32_t header_size, const void *data, uint32_t data_size) {
    while (m_ringbuffer != nullptr && frames != nullptr) {
      uint32_t size = m_ringbuffer->size.load(std::memory_order_acquire);
      if (size >= FRAME_COUNT - 1) {
        uint32_t head = m_ringbuffer->head.load(std::memory_order_acquire);
        Frame *frame = &frames[(head + FRAME_COUNT - 1) % FRAME_COUNT]; // 头部前的位置作为临时插入位置
        frame->flag = 12344321;
        memcpy(frame->header, header, header_size);
        frame->header_size = header_size;
        memcpy(frame->data, data, data_size);
        frame->data_size = data_size;
        uint32_t new_head = (head + 1) % FRAME_COUNT;
        if (m_ringbuffer->head.compare_exchange_strong(head, new_head, std::memory_order_acq_rel, std::memory_order_relaxed)) {
          // printf("MemoryPoolSPMC::write buffer is full\n");
          return;
        }
      } else {
        uint32_t write_pos = (m_ringbuffer->head.load(std::memory_order_acquire) + size) % FRAME_COUNT;
        Frame *frame = &frames[write_pos];
        frame->flag = 12344321;
        memcpy(frame->header, header, header_size);
        frame->header_size = header_size;
        memcpy(frame->data, data, data_size);
        frame->data_size = data_size;
        if (m_ringbuffer->size.compare_exchange_strong(size, size + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) { return; }
      }
      // 如果无法成功写入，再重试
    }
  }
  int read(uint32_t index, void *out_header, void *out_data, uint32_t data_size) {
    if (m_ringbuffer == nullptr || frames == nullptr) {
      printf("MemoryPoolSPMC::read shared memory not attached\n");
      return 0;
    }
    uint32_t head = m_ringbuffer->head.load(std::memory_order_acquire);
    uint32_t size = m_ringbuffer->size.load(std::memory_order_acquire);
    if (size <= 0) {
      // printf("MRAgent::MemoryPoolSPMC::read buffer is empty\n");
      return 0;
    }
    Frame *frame = &frames[head];
    uint32_t read_size = 0;
    if (out_header != nullptr) {
      memcpy(out_header, frame->header, frame->header_size);
      read_size += frame->header_size;
    }
    if (out_data != nullptr && data_size >= frame->data_size) {
      memcpy(out_data, frame->data, frame->data_size);
      read_size += frame->data_size;
    }
    uint32_t new_head = (head + 1) % FRAME_COUNT;
    if (read_size > 0 && m_ringbuffer->head.compare_exchange_strong(head, new_head, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      m_ringbuffer->size.fetch_sub(1, std::memory_order_release);
      return read_size;
    }
    return 0;
  }

  struct alignas(64) MemoryPoolIndex {
    uint64_t flag; // 用于调试，标记是否有效
    std::atomic<uint32_t> head;
    std::atomic<uint32_t> size;
  };
  MemoryPoolIndex *m_ringbuffer = nullptr;

  constexpr static uint32_t FRAME_COUNT = 5;
  constexpr static uint32_t MAX_HEADER_SIZE = 64;
  constexpr static uint32_t FRAME_MAX_DATA_SIZE = (1920 * 1080 * 3);
  constexpr static uint32_t MEMORY_POOL_INDEX_SIZE = sizeof(struct MemoryPoolIndex);

  struct alignas(64) Frame {
    uint64_t flag; // 用于调试，标记是否有效
    uint32_t header_size;
    uint32_t data_size;
    uint32_t width;
    uint32_t height;
    uint8_t header[MAX_HEADER_SIZE];
    uint8_t data[FRAME_MAX_DATA_SIZE];
  };
  Frame *frames = nullptr;
};

class MemoryPoolMPMC {
public:
  void init() {
    m_ringbuffer->enqueue_pos_.store(0, std::memory_order_relaxed);
    m_ringbuffer->dequeue_pos_.store(0, std::memory_order_relaxed);
    for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
      frames[i].flag = 0;
      frames[i].header_size = 0;
      frames[i].data_size = 0;
      frames[i].width = 0;
      frames[i].height = 0;
      frames[i].sequence_.store(i, std::memory_order_relaxed);
    }
  }

  void write(const void *header, uint32_t header_size, const void *data, uint32_t data_size) {
    while (m_ringbuffer != nullptr && frames != nullptr) {
      uint32_t pos = m_ringbuffer->enqueue_pos_.load(std::memory_order_relaxed);
      Frame *frame = &frames[pos & FRAME_MASK];
      uint32_t seq = frame->sequence_.load(std::memory_order_acquire);
      int diff = (int)seq - (int)pos;
      // printf("MemoryPoolMPMC::write pos:%u seq:%u diff:%d\n", pos, seq, diff);
      if (diff == 0) {
        if (m_ringbuffer->enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          frame->flag = 12344321;
          memcpy(frame->header, header, header_size);
          frame->header_size = header_size;
          memcpy(frame->data, data, data_size);
          frame->data_size = data_size;
          frame->sequence_.store(pos + 1, std::memory_order_release);
        }
      } else if (diff < 0) {
        // printf("MemoryPoolMPMCMPMC::write buffer is full\n");
        return; // 满
      }
    }
  }
  int read(uint32_t index, void *out_header, void *out_data, uint32_t data_size) {
    if (m_ringbuffer == nullptr || frames == nullptr) {
      printf("MemoryPoolMPMC::read shared memory not attached\n");
      return 0;
    }
    uint32_t pos = m_ringbuffer->dequeue_pos_.load(std::memory_order_relaxed);
    Frame *frame = &frames[pos & FRAME_MASK];
    uint32_t seq = frame->sequence_.load(std::memory_order_acquire);
    int diff = (int)seq - 1 - (int)pos;
    // printf("MemoryPoolMPMC::read pos:%u seq:%u diff:%d\n", pos, seq, diff);
    if (diff == 0) {
      if (m_ringbuffer->dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        uint32_t read_size = 0;
        if (out_header != nullptr) {
          memcpy(out_header, frame->header, frame->header_size);
          read_size += frame->header_size;
        }
        if (out_data != nullptr && data_size >= frame->data_size) {
          memcpy(out_data, frame->data, frame->data_size);
          read_size += frame->data_size;
        }
        frame->sequence_.store(pos + FRAME_COUNT, std::memory_order_release);
        return read_size;
      }
    }
    return 0; // 读失败
  }

  struct alignas(64) MemoryPoolIndex {
    uint64_t flag; // 用于调试，标记是否有效
    std::atomic<uint32_t> enqueue_pos_;
    std::atomic<uint32_t> dequeue_pos_;
    std::atomic<uint32_t> size;
  };
  MemoryPoolIndex *m_ringbuffer = nullptr;

  constexpr static uint32_t FRAME_COUNT = 8; // 2的幂
  constexpr static uint32_t FRAME_MASK = FRAME_COUNT - 1;
  constexpr static uint32_t MAX_HEADER_SIZE = 64;
  constexpr static uint32_t FRAME_MAX_DATA_SIZE = (1920 * 1080 * 3);
  constexpr static uint32_t MEMORY_POOL_INDEX_SIZE = sizeof(struct MemoryPoolIndex);

  struct alignas(64) Frame {
    // 生产者视角
    // • 生产者拿到一个“逻辑位置” pos，准备往 buffer_[pos & mask_] 里写。
    // • 读该槽位的 sequence_ 值：
    // ‑ 如果 sequence_ == pos → 槽位空闲，可以抢；
    // ‑ 如果 sequence_ <  pos → 槽位已被消费完但尚未被生产者回收，说明 队列已满；
    // ‑ 如果 sequence_ >  pos → 其他生产者正在写，重试。
    // • 成功抢到后，把对象构造到 data_，再把 sequence_ 设为 pos + 1，表示 “我已写完，数据可用”。
    // 消费者视角
    // • 消费者拿到逻辑位置 pos，准备从 buffer_[pos & mask_] 读。
    // • 读该槽位的 sequence_ 值：
    // ‑ 如果 sequence_ == pos + 1 → 数据已就绪，可以抢；
    // ‑ 如果 sequence_ <  pos + 1 → 数据尚未写入，说明 队列为空；
    // ‑ 如果 sequence_ >  pos + 1 → 其他消费者正在读，重试。
    // • 成功抢到后，把对象拷/移到本地变量，再把 sequence_ 设为 pos + capacity_（即下一次生产者应写入时的序号），表示 “我已读完，槽位可回收”。    std::atomic<uint32_t> sequence_;
    uint64_t flag; // 用于调试，标记是否有效
    std::atomic<uint32_t> sequence_;
    uint32_t header_size;
    uint32_t data_size;
    uint32_t width;
    uint32_t height;
    uint8_t header[MAX_HEADER_SIZE];
    uint8_t data[FRAME_MAX_DATA_SIZE];
  };
  Frame *frames = nullptr;
};

#include <cstring>
#include <thread>

using TestPool = MemoryPoolMPMC;

void writer_thread(TestPool &pool, int thread_id) {
  for (int i = 0; i < 100;) {
    char header[TestPool::MAX_HEADER_SIZE];
    char data[100];
    snprintf(header, sizeof(header), "Header-%d-%d", thread_id, i);
    for (int i = 0; i < sizeof(data) - 1; ++i) {
      data[i] = thread_id % 26 + 'A'; // Fill with A-Z
    }
    data[sizeof(data) - 1] = '\0'; // Null-terminate the string
    pool.write(header, strlen(header) + 1, data, sizeof(data));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void reader_thread(TestPool &pool, int thread_id) {
  for (int i = 0; i < 100;) {
    char header[TestPool::MAX_HEADER_SIZE] = {0};
    char data[100] = {0};
    int read_size = pool.read(0, header, data, sizeof(data));
    if (read_size > 0) {
      for (size_t j = 1; j < sizeof(data); ++j) {
        if (data[j] != 0 && data[j] != data[0]) {
          printf("Reader %d error: data inconsistent at index %zu ('%c' != '%c')\n", thread_id, j, data[j], data[0]);
          break;
        }
      }
      // printf("Reader %d read :%s%s\n", thread_id, header, data);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
  }
}

int main() {
  TestPool pool;
  auto *ringbuffer = new TestPool::MemoryPoolIndex();
  auto *frames = new TestPool::Frame[TestPool::FRAME_COUNT];
  pool.m_ringbuffer = ringbuffer;
  pool.frames = frames;
  pool.init();
  constexpr int th_count = 5;
  std::thread writers[th_count], readers[th_count];
  for (int i = 0; i < th_count; ++i) {
    writers[i] = std::thread(writer_thread, std::ref(pool), i);
    readers[i] = std::thread(reader_thread, std::ref(pool), i);
  }
  for (int i = 0; i < th_count; ++i) {
    writers[i].join();
    readers[i].join();
  }

  delete ringbuffer;
  delete[] frames;
  return 0;
}
