//# SPMC
// #include <atomic>
// #include <cstdio>
// #include <cstring>

// class MemoryPoolSPMC {
// public:
//   void init() {}
//   void write(const void *header, uint32_t header_size, const void *data, uint32_t data_size) {
//     while (m_ringbuffer != nullptr && frames != nullptr) {
//       uint32_t size = m_ringbuffer->size.load(std::memory_order_acquire);
//       if (size >= FRAME_COUNT - 1) {
//         uint32_t head = m_ringbuffer->head.load(std::memory_order_acquire);
//         Frame *frame = &frames[(head + FRAME_COUNT - 1) % FRAME_COUNT]; // 头部前的位置作为临时插入位置
//         frame->flag = 12344321;
//         memcpy(frame->header, header, header_size);
//         frame->header_size = header_size;
//         memcpy(frame->data, data, data_size);
//         frame->data_size = data_size;
//         uint32_t new_head = (head + 1) % FRAME_COUNT;
//         if (m_ringbuffer->head.compare_exchange_strong(head, new_head, std::memory_order_acq_rel, std::memory_order_relaxed)) {
//           // printf("MemoryPoolSPMC::write buffer is full\n");
//           return;
//         }
//       } else {
//         uint32_t write_pos = (m_ringbuffer->head.load(std::memory_order_acquire) + size) % FRAME_COUNT;
//         Frame *frame = &frames[write_pos];
//         frame->flag = 12344321;
//         memcpy(frame->header, header, header_size);
//         frame->header_size = header_size;
//         memcpy(frame->data, data, data_size);
//         frame->data_size = data_size;
//         if (m_ringbuffer->size.compare_exchange_strong(size, size + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) { return; }
//       }
//       // 如果无法成功写入，再重试
//     }
//   }
//   int read(uint32_t index, void *out_header, void *out_data, uint32_t data_size) {
//     if (m_ringbuffer == nullptr || frames == nullptr) {
//       printf("MemoryPoolSPMC::read shared memory not attached\n");
//       return 0;
//     }
//     uint32_t head = m_ringbuffer->head.load(std::memory_order_acquire);
//     uint32_t size = m_ringbuffer->size.load(std::memory_order_acquire);
//     if (size <= 0) {
//       // printf("MRAgent::MemoryPoolSPMC::read buffer is empty\n");
//       return 0;
//     }
//     Frame *frame = &frames[head];
//     uint32_t read_size = 0;
//     if (out_header != nullptr) {
//       memcpy(out_header, frame->header, frame->header_size);
//       read_size += frame->header_size;
//     }
//     if (out_data != nullptr && data_size >= frame->data_size) {
//       memcpy(out_data, frame->data, frame->data_size);
//       read_size += frame->data_size;
//     }
//     uint32_t new_head = (head + 1) % FRAME_COUNT;
//     if (read_size > 0 && m_ringbuffer->head.compare_exchange_strong(head, new_head, std::memory_order_acq_rel, std::memory_order_relaxed)) {
//       m_ringbuffer->size.fetch_sub(1, std::memory_order_release);
//       return read_size;
//     }
//     return 0;
//   }

//   struct alignas(64) MemoryPoolIndex {
//     uint64_t flag; // 用于调试，标记是否有效
//     std::atomic<uint32_t> head;
//     std::atomic<uint32_t> size;
//   };
//   MemoryPoolIndex *m_ringbuffer = nullptr;

//   constexpr static uint32_t FRAME_COUNT = 5;
//   constexpr static uint32_t MAX_HEADER_SIZE = 64;
//   constexpr static uint32_t FRAME_MAX_DATA_SIZE = (1920 * 1080 * 3);
//   constexpr static uint32_t MEMORY_POOL_INDEX_SIZE = sizeof(struct MemoryPoolIndex);

//   struct alignas(64) Frame {
//     uint64_t flag; // 用于调试，标记是否有效
//     uint32_t header_size;
//     uint32_t data_size;
//     uint32_t width;
//     uint32_t height;
//     uint8_t header[MAX_HEADER_SIZE];
//     uint8_t data[FRAME_MAX_DATA_SIZE];
//   };
//   Frame *frames = nullptr;
// };

// class MemoryPoolMPMC {
// public:
//   void init() {
//     m_ringbuffer->enqueue_pos_.store(0, std::memory_order_relaxed);
//     m_ringbuffer->dequeue_pos_.store(0, std::memory_order_relaxed);
//     for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
//       frames[i].flag = 0;
//       frames[i].header_size = 0;
//       frames[i].data_size = 0;
//       frames[i].width = 0;
//       frames[i].height = 0;
//       frames[i].sequence_.store(i, std::memory_order_relaxed);
//     }
//   }

//   void write(const void *header, uint32_t header_size, const void *data, uint32_t data_size) {
//     while (m_ringbuffer != nullptr && frames != nullptr) {
//       uint32_t pos = m_ringbuffer->enqueue_pos_.load(std::memory_order_relaxed);
//       Frame *frame = &frames[pos & FRAME_MASK];
//       uint32_t seq = frame->sequence_.load(std::memory_order_acquire);
//       int diff = (int)seq - (int)pos;
//       // printf("MemoryPoolMPMC::write pos:%u seq:%u diff:%d\n", pos, seq, diff);
//       if (diff == 0) {
//         if (m_ringbuffer->enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
//           frame->flag = 12344321;
//           memcpy(frame->header, header, header_size);
//           frame->header_size = header_size;
//           memcpy(frame->data, data, data_size);
//           frame->data_size = data_size;
//           frame->sequence_.store(pos + 1, std::memory_order_release);
//         }
//       } else if (diff < 0) {
//         // printf("MemoryPoolMPMCMPMC::write buffer is full\n");
//         return; // 满
//       }
//     }
//   }
//   int read(uint32_t index, void *out_header, void *out_data, uint32_t data_size) {
//     if (m_ringbuffer == nullptr || frames == nullptr) {
//       printf("MemoryPoolMPMC::read shared memory not attached\n");
//       return 0;
//     }
//     uint32_t pos = m_ringbuffer->dequeue_pos_.load(std::memory_order_relaxed);
//     Frame *frame = &frames[pos & FRAME_MASK];
//     uint32_t seq = frame->sequence_.load(std::memory_order_acquire);
//     int diff = (int)seq - 1 - (int)pos;
//     // printf("MemoryPoolMPMC::read pos:%u seq:%u diff:%d\n", pos, seq, diff);
//     if (diff == 0) {
//       if (m_ringbuffer->dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
//         uint32_t read_size = 0;
//         if (out_header != nullptr) {
//           memcpy(out_header, frame->header, frame->header_size);
//           read_size += frame->header_size;
//         }
//         if (out_data != nullptr && data_size >= frame->data_size) {
//           memcpy(out_data, frame->data, frame->data_size);
//           read_size += frame->data_size;
//         }
//         frame->sequence_.store(pos + FRAME_COUNT, std::memory_order_release);
//         return read_size;
//       }
//     }
//     return 0; // 读失败
//   }

//   struct alignas(64) MemoryPoolIndex {
//     uint64_t flag; // 用于调试，标记是否有效
//     std::atomic<uint32_t> enqueue_pos_;
//     std::atomic<uint32_t> dequeue_pos_;
//     std::atomic<uint32_t> size;
//   };
//   MemoryPoolIndex *m_ringbuffer = nullptr;

//   constexpr static uint32_t FRAME_COUNT = 8; // 2的幂
//   constexpr static uint32_t FRAME_MASK = FRAME_COUNT - 1;
//   constexpr static uint32_t MAX_HEADER_SIZE = 64;
//   constexpr static uint32_t FRAME_MAX_DATA_SIZE = (1920 * 1080 * 3);
//   constexpr static uint32_t MEMORY_POOL_INDEX_SIZE = sizeof(struct MemoryPoolIndex);

//   struct alignas(64) Frame {
//     // 生产者视角
//     // • 生产者拿到一个“逻辑位置” pos，准备往 buffer_[pos & mask_] 里写。
//     // • 读该槽位的 sequence_ 值：
//     // ‑ 如果 sequence_ == pos → 槽位空闲，可以抢；
//     // ‑ 如果 sequence_ <  pos → 槽位已被消费完但尚未被生产者回收，说明 队列已满；
//     // ‑ 如果 sequence_ >  pos → 其他生产者正在写，重试。
//     // • 成功抢到后，把对象构造到 data_，再把 sequence_ 设为 pos + 1，表示 “我已写完，数据可用”。
//     // 消费者视角
//     // • 消费者拿到逻辑位置 pos，准备从 buffer_[pos & mask_] 读。
//     // • 读该槽位的 sequence_ 值：
//     // ‑ 如果 sequence_ == pos + 1 → 数据已就绪，可以抢；
//     // ‑ 如果 sequence_ <  pos + 1 → 数据尚未写入，说明 队列为空；
//     // ‑ 如果 sequence_ >  pos + 1 → 其他消费者正在读，重试。
//     // • 成功抢到后，把对象拷/移到本地变量，再把 sequence_ 设为 pos + capacity_（即下一次生产者应写入时的序号），表示 “我已读完，槽位可回收”。    std::atomic<uint32_t> sequence_;
//     uint64_t flag; // 用于调试，标记是否有效
//     std::atomic<uint32_t> sequence_;
//     uint32_t header_size;
//     uint32_t data_size;
//     uint32_t width;
//     uint32_t height;
//     uint8_t header[MAX_HEADER_SIZE];
//     uint8_t data[FRAME_MAX_DATA_SIZE];
//   };
//   Frame *frames = nullptr;
// };

//# MPMC
// #include <cstring>
// #include <thread>

// using TestPool = MemoryPoolMPMC;

// void writer_thread(TestPool &pool, int thread_id) {
//   for (int i = 0; i < 100;) {
//     char header[TestPool::MAX_HEADER_SIZE];
//     char data[100];
//     snprintf(header, sizeof(header), "Header-%d-%d", thread_id, i);
//     for (int i = 0; i < sizeof(data) - 1; ++i) {
//       data[i] = thread_id % 26 + 'A'; // Fill with A-Z
//     }
//     data[sizeof(data) - 1] = '\0'; // Null-terminate the string
//     pool.write(header, strlen(header) + 1, data, sizeof(data));
//     std::this_thread::sleep_for(std::chrono::milliseconds(10));
//   }
// }

// void reader_thread(TestPool &pool, int thread_id) {
//   for (int i = 0; i < 100;) {
//     char header[TestPool::MAX_HEADER_SIZE] = {0};
//     char data[100] = {0};
//     int read_size = pool.read(0, header, data, sizeof(data));
//     if (read_size > 0) {
//       for (size_t j = 1; j < sizeof(data); ++j) {
//         if (data[j] != 0 && data[j] != data[0]) {
//           printf("Reader %d error: data inconsistent at index %zu ('%c' != '%c')\n", thread_id, j, data[j], data[0]);
//           break;
//         }
//       }
//       // printf("Reader %d read :%s%s\n", thread_id, header, data);
//     }
//     std::this_thread::sleep_for(std::chrono::milliseconds(15));
//   }
// }

// int main() {
//   TestPool pool;
//   auto *ringbuffer = new TestPool::MemoryPoolIndex();
//   auto *frames = new TestPool::Frame[TestPool::FRAME_COUNT];
//   pool.m_ringbuffer = ringbuffer;
//   pool.frames = frames;
//   pool.init();
//   constexpr int th_count = 5;
//   std::thread writers[th_count], readers[th_count];
//   for (int i = 0; i < th_count; ++i) {
//     writers[i] = std::thread(writer_thread, std::ref(pool), i);
//     readers[i] = std::thread(reader_thread, std::ref(pool), i);
//   }
//   for (int i = 0; i < th_count; ++i) {
//     writers[i].join();
//     readers[i].join();
//   }

//   delete ringbuffer;
//   delete[] frames;
//   return 0;
// }

//# 有界阻塞队列
/*#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

class BoundedBlockQueue {
public:
  BoundedBlockQueue(int size) {
    assert(size > 0);
    m_buffer.resize(size);
  }
  void Enqueue(int value) {
    std::unique_lock<std::mutex> lock{m_mtx};
    if (m_size >= m_buffer.size()) {
      m_full_cv.wait(lock, [&]() { return m_size < m_buffer.size(); });
    }
    m_buffer[m_size++] = value;
    m_empty_cv.notify_all();
  }
  int Dequeue() {
    std::unique_lock<std::mutex> lock{m_mtx};
    if (m_size <= 0) {
      m_empty_cv.wait(lock, [&]() { return m_size > 0; });
    }
    int value = m_buffer[m_size--];
    m_full_cv.notify_one();
    return value;
  }

private:
  std::vector<int> m_buffer;
  int m_size{0};
  std::mutex m_mtx;
  std::condition_variable m_empty_cv;
  std::condition_variable m_full_cv;
};

int main() {
  BoundedBlockQueue queue(3);
  std::thread th1{[&]() {
    printf("th1 enqueue 1\n");
    queue.Enqueue(1);
    std::this_thread::sleep_for(std::chrono::seconds{2});
    printf("th1 dequeue 1\n");
    queue.Dequeue();
    printf("th1 completed\n");
  }};
  std::thread th2{[&]() {
    printf("th2 sleep 1s\n");
    queue.Dequeue();
    printf("th2 sleep 2s\n");
    std::this_thread::sleep_for(std::chrono::seconds{2});
    printf("th2 enqueue 1\n");
    queue.Enqueue(1);
    printf("th2 completed\n");
  }};
  th1.join();
  th2.join();
}*/

//# 线程按顺序执行
/*#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

int main() {
  std::mutex mtx;
  std::condition_variable cv1, cv2;
  int status1{-1}, status2{-1};
  std::thread th1{[&]() {
    std::unique_lock<std::mutex> lock{mtx};
    printf("th1 started\n");
    cv1.wait(lock, [&]() { return status1 == 1; });
    printf("th1 completed\n");
  }};
  std::thread th2{[&]() {
    std::unique_lock<std::mutex> lock{mtx};
    printf("th2 started\n");
    cv2.wait(lock, [&]() { return status2 == 1; });
    status1 = 1;
    printf("th2 completed\n");
    cv1.notify_all();
  }};
  std::thread th3{[&]() {
    std::unique_lock<std::mutex> lock{mtx};
    printf("th3 started\n");
    std::this_thread::sleep_for(std::chrono::seconds{2});
    status2 = 1;
    printf("th3 completed\n");
    cv2.notify_all();
  }};
  th1.join();
  th2.join();
  th3.join();
}*/

//# 合并链表
/*#include <cstdio>
#include <initializer_list>
#include <memory>

struct Node {
  std::shared_ptr<Node> next{nullptr};
  int value;
};
struct List {
  std::shared_ptr<Node> head{nullptr};
  int size{0};
};

void merge_list(const List &list1, const List &list2, List &result) {
  if (list1.size == 0 && list2.size == 0) { return; }
  auto node1 = list1.head;
  auto node2 = list2.head;
  auto element = result.head;
  while (element != nullptr && element->next != nullptr) { element = element->next; }
  while (node1 != nullptr || node2 != nullptr) {
    auto node = node1;
    if (node1 == nullptr) {
      node = node2;
      node2 = node2->next;
    } else if (node2 == nullptr) {
      node = node1;
      node1 = node1->next;
    } else if (node1->value < node2->value) {
      node = node1;
      node1 = node1->next;
    } else {
      node = node2;
      node2 = node2->next;
    }
    if (element == nullptr) {
      element = std::make_shared<Node>();
      element->next = nullptr;
      element->value = node->value;
      result.head = element;
      result.size = 1;
    } else {
      element->next = std::make_shared<Node>();
      element = element->next;
      element->next = nullptr;
      element->value = node->value;
      result.size++;
    }
  }
}

void make_list(List &result, const std::initializer_list<int> &values) {
  auto element = result.head;
  for (auto value : values) {
    if (element == nullptr) {
      element = std::make_shared<Node>();
      element->next = nullptr;
      element->value = value;
      result.head = element;
      result.size = 1;
    } else {
      element->next = std::make_shared<Node>();
      element = element->next;
      element->next = nullptr;
      element->value = value;
      result.size++;
    }
  }
}

void print_list(const List &list) {
  auto element = list.head;
  while (element != nullptr) {
    printf("%d ", element->value);
    element = element->next;
  }
  printf("\n");
}

int main() {
  List list1, list2, result;
  make_list(list1, {1, 3, 5, 7});
  make_list(list2, {2, 4, 6, 8});
  merge_list(list1, list2, result);
  print_list(result);
  return 0;
}*/

//# 中序遍历树
/*#include <cassert>
#include <cstdio>
#include <deque>
#include <memory>

struct TreeNode {
  std::shared_ptr<TreeNode> left{nullptr};
  std::shared_ptr<TreeNode> right{nullptr};
  std::shared_ptr<TreeNode> parent{nullptr};
  int value{-1};
};

struct Tree {
  std::shared_ptr<TreeNode> root{nullptr};
  int size{0};
};

void print_treeNode(std::deque<std::shared_ptr<TreeNode>> &deque, int &level, int size) {
  if (size == 0) { return; }
  auto node = deque.front();
  if (node == nullptr) {
    printf("_ ");
  } else {
    printf("%d ", node->value);
    size--;
  }
  deque.pop_front();
  int split_value = 0;
  for (int i = 0; level >= split_value; i++) {
    split_value += (1 << i);
    if (level == split_value) { printf("\n"); }
  }
  if (node != nullptr) {
    deque.push_back(node->left);
    deque.push_back(node->right);
  }
  print_treeNode(deque, ++level, size);
}

void print_tree(const Tree &tree) {
  auto treeNode = tree.root;
  if (treeNode != nullptr) {
    std::deque<std::shared_ptr<TreeNode>> deque;
    deque.push_back(treeNode);
    int level = 0;
    print_treeNode(deque, ++level, tree.size);
    printf("\n");
  }
}

void make_treeNode(std::deque<std::shared_ptr<TreeNode>> &deque, const std::initializer_list<int>::iterator values, const int size, int &pos, int &real_size) {
  if (pos >= size) {
    auto node = deque.front();
    while (!deque.empty()) {
      node = deque.front();
      deque.pop_front();
      auto parent = node->parent;
      if (parent != nullptr) {
        if (parent->left == node) {
          parent->left = nullptr;
        } else {
          parent->right = nullptr;
        }
      }
    }
    return;
  }
  auto node = deque.front();
  deque.pop_front();
  if (values[pos] < 0) {
    auto parent = node->parent;
    if (parent != nullptr) {
      if (parent->left == node) {
        parent->left = nullptr;
      } else {
        parent->right = nullptr;
      }
    }
  } else {
    node->value = values[pos];
    real_size++;
    node->left = std::make_shared<TreeNode>();
    node->left->parent = node;
    deque.push_back(node->left);
    node->right = std::make_shared<TreeNode>();
    node->right->parent = node;
    deque.push_back(node->right);
  }
  make_treeNode(deque, values, size, ++pos, real_size);
}

void make_tree(Tree &tree, const std::initializer_list<int> &values) {
  if (values.size() == 0) { return; }
  if (values.begin()[0] >= 0) {
    tree.root = std::make_shared<TreeNode>();
    std::deque<std::shared_ptr<TreeNode>> deque;
    deque.push_back(tree.root);
    int pos = 0;
    int real_size = 0;
    make_treeNode(deque, values.begin(), values.size(), pos, real_size);
    tree.size = real_size;
  }
}

int main() {
  Tree tree;
  make_tree(tree, {1, 2, 3, -1, -1, 4, 5});
  print_tree(tree);
}*/

//# 最长公共前缀
/*#include <cstdio>
#include <string>
#include <vector>

std::string get_max_prefix(const std::vector<std::string> &strings) {
  if (strings.empty()) { return ""; }
  std::string prefix = strings[0].substr(0, 1);
  int status = -1;
  while (status < 0) {
    for (int i = 0; i < strings.size(); ++i) {
      if (strings[i].find(prefix) != 0) {
        status = 1;
        break;
      } else if (strings[i].size() == prefix.size()) {
        status = 0;
        break;
      }
    }
    if (status < 0) { prefix = strings[0].substr(0, prefix.size() + 1); }
  }
  if (status == 0) { return prefix; }
  return prefix.substr(0, prefix.size() - 1);
}

int main() {
  std::vector<std::string> strings{"1234", "123456", "12348"};
  auto prefix = get_max_prefix(strings);
  printf("%s\n", prefix.c_str());
}*/

//# 反转字符串中的单词
// #include <cassert>
// #include <cstdio>
// #include <string>

// std::string reverse_string(const std::string &in_str, int pos_begin, int pos_end) {
//   assert(pos_begin >= 0);
//   assert(pos_end < in_str.size());
//   std::string rev_str;
//   for (int pos = pos_end; pos >= pos_begin; --pos) { rev_str += in_str[pos]; }
//   return rev_str;
// }

// std::string reverse_string(const std::string &in_str) {
//   auto result = reverse_string(in_str, 0, in_str.size() - 1);
//   std::string rev_str;
//   int pos_begin = 0, pos_end = 0;
//   while (pos_begin < in_str.size()) {
//     pos_end = result.find_first_of(" \t", pos_begin);
//     if (pos_end > pos_begin) {
//       rev_str += reverse_string(result, pos_begin, pos_end - 1);
//       rev_str += result[pos_end];
//       pos_begin = pos_end + 1;
//     } else if (pos_end == std::string::npos) {
//       rev_str += result.substr(pos_begin);
//       pos_begin = result.size();
//     } else {
//       rev_str += result[pos_begin];
//       pos_begin++;
//     }
//   }
//   return rev_str;
// }

// int main() {
//   std::string str = "I am a student   ";
//   printf("%s\n", reverse_string(str).c_str());
//   return 0;
// }

//# 有界队列
// #include <vector>
// #include <cstdio>
// #include <sstream>
// #include <cstdint>

// template<typename T>
// class MyQueue {
// private:
// std::vector<T> _buffer;
// uint32_t _head{0};
// uint32_t _tail{0};
// uint32_t _capacity;
// public:
// explicit MyQueue(uint32_t capacity): _buffer(capacity + 1), _capacity(capacity + 1) {}
// bool empty() {
// return _head == _tail;
// }
// bool full() {
// return (_tail + 1) % _capacity == _head;
// }
// bool push_back(const T& val) {
// if (full()) {
// printf("full\n");
// return false;
// }
// _buffer[_tail] = val;
// _tail = (_tail + 1) % _capacity;
// return true;
// }
// bool pop_front(T& val) {
// if (empty()) {
// printf("empty\n");
// return false;
// }
// val = _buffer[_head];
// _head = (_head + 1) % _capacity;
// return true;
// }
// std::string debug() {
// std::stringstream ss;
// ss << "_buffer [" << _buffer.size() << "] { ";
// for (auto &i : _buffer) {
// ss << i << " ";
// }
// ss << "} _head [" << _head << "] " << "_tail [" << _tail << "] ";
// return ss.str();
// }
// };

// int test_myqueue() {
// MyQueue<uint32_t> myQueue(10);
// printf("1\n");
// printf("debug: %s\n", myQueue.debug().c_str());
// printf("empty: %d\n", myQueue.empty());
// printf("full: %d\n", myQueue.full());
// printf("2\n");
// myQueue.push_back(1);
// printf("debug: %s\n", myQueue.debug().c_str());
// uint32_t val = 0;
// myQueue.pop_front(val);
// printf("debug: %s\n", myQueue.debug().c_str());
// printf("empty: %d\n", myQueue.empty());
// printf("full: %d\n", myQueue.full());
// myQueue.pop_front(val);
// printf("3\n");
// for (auto i = 0; i < 14; ++i) {
// myQueue.push_back(1);
// printf("debug: %s\n", myQueue.debug().c_str());
// }
// printf("4\n");
// for (auto i = 0; i < 14; ++i) {
// myQueue.pop_front(val);
// printf("debug: %s\n", myQueue.debug().c_str());
// }
// return 0;
// }

//# 反转字符串中的单词
// #include <string>
// auto & npos = std::string::npos;
// bool reverse_string(std::string & in_str, size_t start_pos, size_t end_pos) {
// if (start_pos < 0) return false;
// if (start_pos == npos || end_pos == npos) return false;
// if (start_pos >= end_pos) return false;
// auto len = end_pos - start_pos;
// auto start_elem = &in_str[start_pos];
// auto end_elem = &in_str[end_pos];
// for (auto i = 0; i < len / 2; ++i) {
// auto val = start_elem[i];
// start_elem[i] = end_elem[- 1 - i];
// end_elem[- 1 - i] = val;
// }
// return true;
// }
// void reverse_full_string(std::string & in_str) {
// reverse_string(in_str, 0, in_str.length());
// size_t start_pos = 0;
// size_t end_pos = in_str.find(' ', start_pos);
// auto res = end_pos != npos;
// while(res) {
// res = reverse_string(in_str, start_pos, end_pos);
// if (res) {
// start_pos = end_pos + 1;
// end_pos = in_str.find(' ', start_pos);
// res = end_pos != npos;
// }
// }
// if (start_pos < in_str.length()) {
// reverse_string(in_str, start_pos, in_str.length());
// }
// }
// int test_reverse_string() {
// std::string str = "123";
// reverse_string(str, 0, str.length());
// printf("%s\n", str.c_str());
// str = "123 456";
// reverse_full_string(str);
// printf("%s\n", str.c_str());
// str = "123 456 7 890";
// reverse_full_string(str);
// printf("%s\n", str.c_str());
// return 0;
// }

//# 合并有序数组
// #include <algorithm>
// #include <iostream>
// template<typename T>
// void print_vector(const std::vector<T>& v) {
// std::for_each(v.begin(), v.end(), [](const T& val) {
// std::cout << val << " ";
// });
// std::cout << std::endl;
// }
// #include <vector>
// template<typename T>
// std::vector<T> merge(const std::vector<T>& v1, const std::vector<T>& v2) {
// std::vector<T> v;
// auto pos1 = 0;
// auto pos2 = 0;
// for (; pos1 < v1.size() && pos2 < v2.size();) {
// if (v1[pos1] <= v2[pos2]) {
// v.emplace_back(v1[pos1++]);
// } else {
// v.emplace_back(v2[pos2++]);
// }
// }
// if (pos1 < v1.size()) {
// v.insert(v.end(), v1.begin() + pos1, v1.end());
// }
// if (pos2 < v2.size()) {
// v.insert(v.end(), v2.begin() + pos2, v2.end());
// }
// return v;
// }
// template<typename T>
// std::vector<T> merge_new(const std::vector<T>& in_v, T in_val) {
// std::vector<T> v;
// std::vector<size_t> pos_v;
// for (auto i = 0; i < in_v.size(); ++i) {
// if (in_v[i] < in_val) {
// v.emplace_back(in_v[i]);
// } else {
// pos_v.emplace_back(i);
// }
// }
// for (auto i = 0; i < pos_v.size(); ++i) {
// v.emplace_back(in_v[pos_v[i]]);
// }
// return v;
// }
// int test_merge() {
// {
// std::vector<uint32_t> v1{1,2,3};
// std::vector<uint32_t> v2{4,5,6};
// auto v = merge<uint32_t>(v1, v2);
// print_vector<uint32_t>(v);
// }
// {
// std::vector<uint32_t> v1{1,2,3};
// std::vector<uint32_t> v2{1,2,3};
// auto v = merge<uint32_t>(v1, v2);
// print_vector<uint32_t>(v);
// }
// {
// std::vector<uint32_t> v{1,4,3,2,5,2};
// auto v_o = merge_new<uint32_t>(v, 3);
// print_vector<uint32_t>(v_o);
// }
// return 0;
// }

//# 合并链表、反转链表、旋转链表
// #include <memory>
// template<typename T>
// class List;
// template<typename T>
// class Node {
// private:
// std::shared_ptr<Node> _prev{nullptr};
// std::shared_ptr<Node> _next{nullptr};
// T _value;
// friend class List<T>;
// };
// template<typename T>
// class List {
// private:
// std::shared_ptr<Node<T>> _head{nullptr};
// std::shared_ptr<Node<T>> _tail{nullptr};
// size_t _size{0};
// public:
// size_t size() {
// return _size;
// }
// void push_back(const T& val) {
// auto new_node = std::make_shared<Node<T>>();
// new_node->_value = val;
// if (_size == 0) {
// _head = _tail = new_node;
// } else {
// _tail->_next = new_node;
// new_node->_prev = _tail;
// _tail = new_node;
// }
// _size++;
// }
// bool pop_front(T& val) {
// if (_size == 0) return false;
// val = _head->_value;
// _head = _head->_next;
// if (_head == nullptr) {
// _tail = _head;
// } else {
// _head->_prev = nullptr;
// }
// _size--;
// return true;
// }
// void reverse() {
// auto node = std::make_shared<Node<T>>();
// auto e = _head;
// for (; e != nullptr;) {
// node = e->_prev;
// e->_prev = e->_next;
// e->_next = node;
// e = e->_prev;
// }
// node = _head;
// _head = _tail;
// _tail = node;
// }
// bool reverse_range(size_t begin, size_t end) {
// if (begin < 0) {
// printf("bad begin < 0\n");
// }
// if (begin > _size - 1) {
// printf("bad begin > _size - 1\n");
// }
// if (end < 0) {
// printf("bad end < 0\n");
// }
// if (end > _size - 1) {
// printf("bad end > _size - 1\n");
// }
// if (end < begin) {
// printf("bad end < begin\n");
// }
// auto e_b = _head;
// for (auto i = 0; i < begin; ++i) {
// e_b = e_b->_next;
// }
// auto e_e = e_b;
// for (auto i = begin; i < end; ++i) {
// e_e = e_e->_next;
// }
// auto head = e_b->_prev;
// auto tail = e_e->_next;
// auto node = std::make_shared<Node<T>>();
// for (auto e = e_b; e != tail;) {
// node = e->_next;
// e->_next = e->_prev;
// e->_prev = node;
// e = node;
// }
// if (head != nullptr) {
// head->_next = e_e;
// }
// e_e->_prev = head;
// if (tail != nullptr) {
// tail->_prev = e_b;
// }
// e_b->_next = tail;
// if (head == nullptr) {
// _head = e_e;
// }
// if (tail == nullptr) {
// _tail = e_b;
// }
// return true;
// }
// std::string debug() {
// std::stringstream ss;
// ss << "[" << _size << "] ";
// for (auto e = _head; e != nullptr; e = e->_next) {
// ss << e->_value << " ";
// }
// return ss.str();
// }
// };
// int test_list() {
// List<uint32_t> list;

// printf("%s\n", list.debug().c_str());
// list.push_back(1);
// list.push_back(2);
// list.push_back(3);
// printf("%s\n", list.debug().c_str());
// uint32_t val = 0;
// list.pop_front(val);
// printf("%s\n", list.debug().c_str());
// list.pop_front(val);
// printf("%s\n", list.debug().c_str());
// list.pop_front(val);
// printf("%s\n", list.debug().c_str());
// list.push_back(1);
// list.push_back(2);
// list.push_back(3);
// list.reverse();
// printf("%s\n", list.debug().c_str());
// list.push_back(4);
// list.push_back(5);
// list.push_back(6);
// printf("%s\n", list.debug().c_str());

// list.push_back(1);
// list.push_back(2);
// list.push_back(3);
// printf("%s\n", list.debug().c_str());
// list.reverse_range(0, 1);
// printf("%s\n", list.debug().c_str());
// list.reverse_range(0, list.size() - 1);
// printf("%s\n", list.debug().c_str());
// list.reverse_range(1, list.size() - 1);
// printf("%s\n", list.debug().c_str());
// return 0;
// }

// #include <algorithm>
// size_t get_max_value(size_t in_value) {
// auto in_value_str = std::to_string(in_value);
// auto max_it = std::max_element(in_value_str.begin(), in_value_str.end());
// auto max_value = in_value;
// if (max_it != in_value_str.begin()) {
// auto v = *in_value_str.begin();
// *in_value_str.begin() = *max_it;
// *max_it = v;
// max_value = std::atol(in_value_str.c_str());
// }
// return max_value;
// }
// int test_get_max_value() {
// printf("%ld\n", get_max_value(4321));
// return 0;
// }

//# BlockingCircularBuffer
// #include <vector>
// #include <mutex>
// #include <condition_variable>
// #include <pthread.h>
// template <typename T>
// class BlockingCircularBuffer {
// public:
// explicit BlockingCircularBuffer(std::size_t capacity) : _buffer(capacity + 1), _capacity(capacity + 1) {}

// void Push(const T &v) {
// std::unique_lock<std::mutex> lock(_mut);
// if (Full()) {
// printf("full %lu\n", pthread_self());
// _cv1.wait(lock, [this] { return !Full(); });
// }
// _buffer[_tail] = v;
// _tail = (_tail + 1) % _capacity;
// _cv2.notify_all();
// }
// T Take() {
// T v;
// std::unique_lock<std::mutex> lock(_mut);
// if (Empty()) {
// printf("empty %lu\n", pthread_self());
// _cv2.wait(lock, [this] { return !Empty(); });
// }
// v = _buffer[_head];
// _head = (_head + 1) % _capacity;
// _cv1.notify_all();
// return v;
// }
// bool Full() {
// return (_tail + 1) % _capacity == _head;
// }
// bool Empty() {
// return _head == _tail;
// }
// private:
// std::vector<T> _buffer;
// std::size_t _head{0};
// std::size_t _tail{0};
// std::size_t _capacity{0};
// std::mutex _mut;
// std::condition_variable _cv1;
// std::condition_variable _cv2;
// };
// #include <chrono>
// #include <thread>
// int test_bcb() {
// BlockingCircularBuffer<uint32_t> bcb(2);
// std::thread t1([&bcb] {
// bcb.Push(1);
// printf("push 1\n");
// bcb.Push(2);
// printf("push 2\n");
// bcb.Push(3);
// printf("push 3\n");
// printf("push complete\n");
// });
// std::this_thread::sleep_for(std::chrono::seconds(5));
// printf("sleep complete\n");
// std::thread t2([&bcb] {
// bcb.Take();
// printf("take 1\n");
// bcb.Take();
// printf("take 2\n");
// bcb.Take();
// printf("take 3\n");
// bcb.Take();
// printf("take 4\n");
// printf("take complete\n");
// });
// t1.join();
// printf("t1 joinn");
// bcb.Push(0);
// t2.join();
// printf("t2 join\n");
// return 0;
// }

// int main() {
// test_bcb();

// char key = 0;
// printf("press key to continue\n");
// std::cin >> key;

// std::vector<std::string> names = {"hi", "test", "foo"};
// std::vector<std::size_t> name_sizes;
// std::transform(names.begin(), names.end(), std::back_inserter(name_sizes), [](const std::string &name) {
// return name.size();
// });
// print_vector<std::size_t>(name_sizes);
// }

// int func(int *arr, int len) {
// int area = 0;
// for (int *p = arr, i = 0; i < len - 1; ++i) {
// for (int j = i + 1, *q = p + j; j < len; ++j) {
// int height = std::min(*p, *q);
// int width = j - i;
// int val = width * height;
// if (val > area) {
// area = val;
// }
// ++q;
// }
// ++p;
// }
// return area;
// }

//# 反转字符串中的单词
// #include <cstdio>
// #include <string>

// std::string reverse_string(const std::string &in_str, int begin_pos, int end_pos) {
// std::string ret_str;
// for (int pos = end_pos; pos >= begin_pos; --pos) {
// ret_str += in_str[pos];
// }
// return ret_str;
// }

// std::string reverse_string(const std::string &in_str) {
// if (in_str.empty()) return "";
// std::string tmp_str = reverse_string(in_str, 0, in_str.size() - 1);
// int begin_pos = 0, end_pos = -1;
// std::string ret_str;
// while (begin_pos < tmp_str.size()) {
// end_pos = tmp_str.find_first_of(" \t", begin_pos);
// if (end_pos < 0) {
// ret_str += reverse_string(tmp_str, begin_pos, tmp_str.size() - 1);
// break;
// } else if (end_pos == begin_pos) {
// ret_str += tmp_str[begin_pos];
// ++begin_pos;
// } else {
// ret_str += reverse_string(tmp_str, begin_pos, end_pos - 1);
// ret_str += tmp_str[end_pos];
// begin_pos = end_pos + 1;
// }
// }
// return ret_str;
// }

// int main() {
// std::string in_str = " I am a student";
// printf("%s\n", reverse_string(in_str).c_str());
// return 0;
// }

//# 接雨水
// #include <cstdio>
// #include <vector>

// int get_value(const std::vector<int> &rain_vec) {
// int ret_value{0};
// bool stop{false};
// int level{1};
// while(!stop) {
// stop = true;
// int flag{0};
// int sub_value{0};
// for(int i = 1; i < rain_vec.size(); ++i) {
// if (rain_vec[i] >= level) {
// stop = false;
// }
// if (flag > 0) {
// if (rain_vec[i] < level) {
// ++sub_value;
// } else {
// flag = 0;
// ret_value += sub_value;
// sub_value = 0;
// }
// } else {
// if (rain_vec[i-1] >= level && rain_vec[i] < level) {
// flag = 1;
// ++sub_value;
// }
// }
// }
// ++level;
// }
// return ret_value;
// }

// int main() {
// std::vector<int> rain_vec{0,1,0,2,1,0,1,3,2,1,2,1};
// // std::vector<int> rain_vec{4,2,0,3,2,5};
// printf("%d\n", get_value(rain_vec));
// return 0;
// }

//# 整数转罗马数字
// #include <cstdio>
// #include <string>

// std::string convert(int value) {
// if (value >= 4000) return "";
// std::string ret_str;
// int k_val = value / 1000;
// for (int i = 0; i < k_val;++i) ret_str += "M";
// int h_val = (value / 100) % 10;
// switch (h_val) {
// case 1:
// {
// ret_str +="C";
// break;
// }
// case 2:
// {
// ret_str +="CC";
// break;
// }
// case 3:
// {
// ret_str +="CCC";
// break;
// }
// case 4:
// {
// ret_str +="CD";
// break;
// }
// case 5:
// {
// ret_str +="D";
// break;
// }
// case 6:
// {
// ret_str +="DC";
// break;
// }
// case 7:
// {
// ret_str +="DCC";
// break;
// }
// case 8:
// {
// ret_str +="DCCC";
// break;
// }
// case 9:
// {
// ret_str +="CM";
// break;
// }
// }
// int t_val = (value / 10) % 10;
// switch (t_val) {
// case 1:
// {
// ret_str +="X";
// break;
// }
// case 2:
// {
// ret_str +="XX";
// break;
// }
// case 3:
// {
// ret_str +="XXX";
// break;
// }
// case 4:
// {
// ret_str +="XL";
// break;
// }
// case 5:
// {
// ret_str +="L";
// break;
// }
// case 6:
// {
// ret_str +="LX";
// break;
// }
// case 7:
// {
// ret_str +="LXX";
// break;
// }
// case 8:
// {
// ret_str +="LXXX";
// break;
// }
// case 9:
// {
// ret_str +="XC";
// break;
// }
// }
// int n_val = value % 10;
// switch (n_val) {
// case 1:
// {
// ret_str +="I";
// break;
// }
// case 2:
// {
// ret_str +="II";
// break;
// }
// case 3:
// {
// ret_str +="III";
// break;
// }
// case 4:
// {
// ret_str +="IV";
// break;
// }
// case 5:
// {
// ret_str +="V";
// break;
// }
// case 6:
// {
// ret_str +="VI";
// break;
// }
// case 7:
// {
// ret_str +="VII";
// break;
// }
// case 8:
// {
// ret_str +="VIII";
// break;
// }
// case 9:
// {
// ret_str +="IX";
// break;
// }
// }
// return ret_str;
// }

//# 罗马数字转整数
// #include <map>
// int convert(const std::string in_str) {
// int ret_val{0};
// std::map<std::string, int> convert_map;
// convert_map.insert({"I", 1});
// convert_map.insert({"V", 5});
// convert_map.insert({"X", 10});
// convert_map.insert({"L", 50});
// convert_map.insert({"C", 100});
// convert_map.insert({"D", 500});
// convert_map.insert({"M", 1000});
// int prev_val{0};
// for (int i = 0; i < in_str.size(); ++i) {
// if (convert_map.count(in_str.substr(i, 1)) > 0) {
// int val = convert_map[in_str.substr(i, 1)];
// switch(val) {
// case 1:
// {
// ret_val += prev_val;
// prev_val = 1;
// break;
// }
// case 5:
// {
// if (prev_val == 1) {
// ret_val += 4;
// } else {
// ret_val += 5;
// }
// prev_val = 0;
// break;
// }
// case 10:
// {
// if (prev_val == 1) {
// ret_val += 9;
// prev_val = 0;
// } else {
// ret_val += prev_val;
// prev_val = 10;
// }
// break;
// }
// case 50:
// {
// if (prev_val == 10) {
// ret_val += 40;
// } else {
// ret_val += 50;
// }
// prev_val = 0;
// break;
// }
// case 100:
// {
// if (prev_val == 10) {
// ret_val += 90;
// prev_val = 0;
// } else {
// ret_val += prev_val;
// prev_val = 100;
// }
// break;
// }
// case 500:
// {
// if (prev_val == 100) {
// ret_val += 400;
// } else {
// ret_val += 500;
// }
// break;
// }
// case 1000:
// {
// if (prev_val == 100) {
// ret_val += 900;
// } else {
// ret_val += 1000;
// }
// prev_val = 0;
// break;
// }
// }
// }
// }
// ret_val += prev_val;
// return ret_val;
// }
// int main() {
// // printf("%s\n", convert(3749).c_str());
// // printf("%s\n", convert(365).c_str());
// // printf("%d\n", convert("III"));
// // printf("%d\n", convert("IV"));
// // printf("%d\n", convert("IX"));
// // printf("%d\n", convert("LVIII"));
// printf("%d\n", convert("MCMXCIV"));
// return 0;
// }

//# 长度最小的子数组
// #include <cstdio>
// #include <vector>

// int search_sub_array(const std::vector<int> &in_vec, int value) {
// int ret_value{0};
// for (int i = 0; i < in_vec.size(); ++i) {
// if (in_vec[i] >= value) {
// ret_value = 1;
// break;
// }
// int pos = i + 1;
// while(pos < in_vec.size()) {
// int sum_value{in_vec[i]};
// int count{2};
// for (int j = pos; j < in_vec.size(); ++j) {
// if (sum_value + in_vec[j] >= value) {
// if (ret_value == 0) {
// ret_value = count;
// break;
// } else if (count < ret_value) {
// ret_value = count;
// break;
// }
// }
// sum_value += in_vec[j];
// ++count;
// }
// ++pos;
// }
// }
// return ret_value;
// }

// int main() {
// {
// std::vector<int> array{2,3,1,2,4,3};
// printf("%d\n", search_sub_array(array, 7));
// }
// {
// std::vector<int> array{1,4,4};
// printf("%d\n", search_sub_array(array, 4));
// }
// {
// std::vector<int> array{1,1,1,1,1,1,1,1};
// printf("%d\n", search_sub_array(array, 11));
// }
// return 0;
// }

//# 判断子序列
// #include <cstdio>
// #include <string>

// bool check_sub_str(const std::string &in_str, const std::string &in_sub_str) {
// if (in_str.empty() || in_sub_str.empty()) return false;
// int pos{0};
// int status{0};
// for (int i = 0; i < in_sub_str.size();) {
// status = 0;
// for (int j = pos; j < in_str.size(); ++j) {
// if (in_str[j] == in_sub_str[i]) {
// ++i;
// pos = j+1;
// status = 1;
// break;
// }
// }
// if (status == 0) {
// break;
// }
// }
// return status != 0;
// }

// int main() {
// {
// printf("%d\n", check_sub_str("ahbgdc", "abc"));
// printf("%d\n", check_sub_str("ahbgdc", "axc"));
// }
// return 0;
// }

//# 反转链表、旋转链表
// #include <cstdio>
// #include <memory>
// #include <initializer_list>

// struct Node {
// std::shared_ptr<Node> next{nullptr};
// int value{-1};
// };

// struct List {
// std::shared_ptr<Node> head{nullptr};
// unsigned int size{0};
// };

// List reverse_list(const List &in_list) {
// if (in_list.size == 0) return List{};
// List ret_list;
// auto pnode = std::make_shared<Node>();
// pnode->value = in_list.head->value;
// ret_list.head = pnode;
// ret_list.size = 1;
// pnode = in_list.head->next;
// while (pnode != nullptr) {
// auto new_node = std::make_shared<Node>();
// new_node->value = pnode->value;
// new_node->next = ret_list.head;
// ret_list.head = new_node;
// ++ret_list.size;
// pnode = pnode->next;
// }
// return ret_list;
// }

// List reverse_list(const List &in_list, int left, int right) {
// if (left < 0 || right < 0 || left >= in_list.size || right >= in_list.size || left > right) return List{};
// List ret_list;
// auto phead = std::make_shared<Node>();
// auto pnode = in_list.head;
// phead->value = pnode->value;
// ret_list.head = phead;
// ++ret_list.size;
// for (int i = 1; i < left; ++i) {
// pnode = pnode->next;
// phead->next = std::make_shared<Node>();
// phead = phead->next;
// phead->value = pnode->value;
// ++ret_list.size;
// }
// auto plast = phead->next;
// auto pnewnode = phead->next;
// for (int i = left; i <= right; ++i) {
// pnode = pnode->next;
// phead->next = std::make_shared<Node>();
// phead->next->value = pnode->value;
// phead->next->next = pnewnode;
// pnewnode = phead->next;
// ++ret_list.size;
// if (plast == nullptr) plast = pnewnode;
// }
// phead = plast;
// for (int i = right + 1; i < in_list.size; ++i) {
// pnode = pnode->next;
// phead->next = std::make_shared<Node>();
// phead = phead->next;
// phead->value = pnode->value;
// ++ret_list.size;
// }
// return ret_list;
// }

// List rotate_list(const List & in_list, int step) {
// if (in_list.size == 0) return List{};
// step %= in_list.size;
// List ret_list;
// auto phead = std::make_shared<Node>();
// auto pnode = in_list.head;
// phead->value = pnode->value;
// ret_list.head = phead;
// ++ret_list.size;
// int count{1};
// for (int i = 1; i < step; ++i) {
// pnode = pnode->next;
// phead->next = std::make_shared<Node>();
// phead = phead->next;
// phead->value = pnode->value;
// ++ret_list.size;
// ++count;
// }
// if (count < in_list.size) {
// auto pmiddle = ret_list.head;
// if (step > 0) {
// phead = std::make_shared<Node>();
// pnode = pnode->next;
// phead->value = pnode->value;
// ret_list.head = phead;
// ++ret_list.size;
// ++count;
// } else {
// pmiddle = nullptr;
// }
// for (int i = count; i < in_list.size; ++i) {
// pnode = pnode->next;
// phead->next = std::make_shared<Node>();
// phead = phead->next;
// phead->value = pnode->value;
// ++ret_list.size;
// }
// phead->next = pmiddle;
// }
// return ret_list;
// }

// void print_list(const List &in_list) {
// auto pnode = in_list.head;
// while (pnode != nullptr) {
// printf("%d ", pnode->value);
// pnode = pnode->next;
// }
// printf("\n");
// }

// List make_list(const std::initializer_list<int> &in_list) {
// List ret_list;
// if (in_list.size() == 0) return ret_list;
// auto phead = std::make_shared<Node>();
// phead->value = *in_list.begin();
// ret_list.head = phead;
// ++ret_list.size;
// for (int i = 1; i < in_list.size(); ++i) {
// phead->next = std::make_shared<Node>();
// phead = phead->next;
// phead->value = *(in_list.begin() + i);
// ++ret_list.size;
// }
// return ret_list;
// }
// int main() {
// List list = make_list({1,2,3,4,5,6});
// print_list(list);
// list = reverse_list(list);
// print_list(list);
// list = reverse_list(list,2,4);
// print_list(list);
// list = rotate_list(list,0);
// print_list(list);
// list = rotate_list(list,1);
// print_list(list);
// list = rotate_list(list,2);
// print_list(list);
// list = rotate_list(list,5);
// print_list(list);
// return 0;
// }

//# 用最少数量的箭引爆气球
// #include <algorithm>
// #include <cstdio>
// #include <vector>

// int get_arrow_count(const std::vector<std::vector<int>> &in_vec) {
//   if (in_vec.size() == 0)
//     return 0;
//   auto vec{in_vec};
//   std::sort(vec.begin(), vec.end(),
//             [](const auto &e1, const auto &e2) { return e1[0] < e2[0]; });
//   for (int i = 1; i < vec.size();) {
//     if (vec[i][0] <= vec[i - 1][1] && vec[i][1] >= vec[i - 1][0]) {
//       vec[i - 1][0] = std::max(vec[i - 1][0], vec[i][0]);
//       vec[i - 1][1] = std::min(vec[i - 1][1], vec[i][1]);
//       vec.erase(vec.begin() + i);
//     } else {
//       ++i;
//     }
//   }
//   return vec.size();
// }

// int main() {
//   {
//     std::vector<std::vector<int>> bubbles{{10, 16}, {2, 8}, {1, 6}, {7, 12}};
//     printf("%d\n", get_arrow_count(bubbles));
//   }
//   {
//     std::vector<std::vector<int>> bubbles{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
//     printf("%d\n", get_arrow_count(bubbles));
//   }
//   {
//     std::vector<std::vector<int>> bubbles{{1, 2}, {2, 3}, {3, 4}, {4, 5}};
//     printf("%d\n", get_arrow_count(bubbles));
//   }
//   {
//     std::vector<std::vector<int>> bubbles{{1, 2}};
//     printf("%d\n", get_arrow_count(bubbles));
//   }
//   {
//     std::vector<std::vector<int>> bubbles{{1, 2}, {3, 4}};
//     printf("%d\n", get_arrow_count(bubbles));
//   }

//   return 0;
// }

//# 合并区间
// typedef vector<vector<int>> intervals;
// intervals merge_intervals(const intervals &i1, const intervals &i2) {
//   if (i1.empty()) {
//     return i2;
//   } else if (i2.empty()) {
//     return i1;
//   }
//   intervals ret_value{i1};
//   {
//     intervals &i1 = ret_value;
//     int pos1{0}, pos2{0};
//     while (pos1 < i1.size()) {
//       {
//         intervals &i2 = i1;
//         int pos2 = pos1 + 1;
//         if (pos2 < i2.size()) {
//           int x = std::max(i1[pos1][0], i2[pos2][0]);
//           int y = std::min(i1[pos1][1], i2[pos2][1]);
//           if (x <= y) {
//             i1[pos1][0] = std::min(i1[pos1][0], i2[pos2][0]);
//             i1[pos1][1] = std::max(i1[pos1][1], i2[pos2][1]);
//             i2.erase(i2.begin() + pos2);
//             continue;
//           }
//         }
//       }
//       if (pos2 < i2.size()) {
//         int x = std::max(i1[pos1][0], i2[pos2][0]);
//         int y = std::min(i1[pos1][1], i2[pos2][1]);
//         if (x <= y) {
//           i1[pos1][0] = std::min(i1[pos1][0], i2[pos2][0]);
//           i1[pos1][1] = std::max(i1[pos1][1], i2[pos2][1]);
//           ++pos2;
//         } else if (i2[pos2][1] < i1[pos1][0]) {
//           i1.insert(i1.begin() + pos1, i2[pos2]);
//           ++pos1;
//           ++pos2;
//         } else {
//           ++pos1;
//         }
//         continue;
//       }
//       ++pos1;
//     }
//     while (pos2 < i2.size()) {
//       i1.emplace_back(i2[pos2]);
//       ++pos2;
//     }
//   }
//   return ret_value;
// }

// int main() {
//   {
//     intervals i1{{1, 3}, {6, 9}};
//     intervals i2{{2, 5}, {10, 12}};
//     auto merged = merge_intervals(i1, i2);
//     for (const auto &interval : merged) { printf("[%d,%d] ", interval[0], interval[1]); }
//     printf("\n");
//   }
//   {
//     intervals i1{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}};
//     intervals i2{{4, 8}};
//     auto merged = merge_intervals(i1, i2);
//     for (const auto &interval : merged) { printf("[%d,%d] ", interval[0], interval[1]); }
//     printf("\n");
//   }
//   return 0;
// }

//# 插入区间
// #include <algorithm>
// #include <cstdio>
// #include <vector>

// typedef std::vector<std::vector<int>> ranges;
// ranges get_ranges(const ranges &r1, const ranges &r2) {
//   ranges ret_ranges{r1};
//   if (r2.empty())
//     return ret_ranges;
//   if (r1.empty()) {
//     ret_ranges = ranges{r2};
//     return ret_ranges;
//   }
//   int i{1}, j{0};
//   while (i < ret_ranges.size() || j < r2.size()) {
//     if (j < r2.size()) {
//       if (i == ret_ranges.size()) {
//         ret_ranges.emplace_back(r2[j]);
//         ++j;
//       } else if (ret_ranges[i - 1][0] <= r2[j][1] &&
//                  ret_ranges[i - 1][1] >= r2[j][0]) {
//         ret_ranges[i - 1][0] = std::min(ret_ranges[i - 1][0], r2[j][0]);
//         ret_ranges[i - 1][1] = std::max(ret_ranges[i - 1][1], r2[j][1]);
//         ++j;
//       } else if (ret_ranges[i - 1][0] > r2[j][1]) {
//         ret_ranges.insert(ret_ranges.begin() + i - 1, r2[j]);
//         ++i;
//         ++j;
//       }
//     }
//     if (i < ret_ranges.size()) {
//       if (ret_ranges[i][0] <= ret_ranges[i - 1][1] &&
//           ret_ranges[i][1] >= ret_ranges[i - 1][0]) {
//         ret_ranges[i - 1][0] = std::min(ret_ranges[i - 1][0],
//         ret_ranges[i][0]); ret_ranges[i - 1][1] = std::max(ret_ranges[i -
//         1][1], ret_ranges[i][1]); ret_ranges.erase(ret_ranges.begin() + i);
//       } else {
//         ++i;
//       }
//     }
//   }
//   return ret_ranges;
// }

// int main() {
//   {
//     ranges r1{{1, 3}, {6, 9}};
//     ranges r2{{2, 5}};
//     auto r = get_ranges(r1, r2);
//     std::for_each(r.begin(), r.end(),
//                   [](const auto &e) { printf("[%d,%d]", e[0], e[1]); });
//     printf("\n");
//   }
//   {
//     ranges r1{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}};
//     ranges r2{{4, 8}};
//     auto r = get_ranges(r1, r2);
//     std::for_each(r.begin(), r.end(),
//                   [](const auto &e) { printf("[%d,%d]", e[0], e[1]); });
//     printf("\n");
//   }
// }

// 盛最多水的容器
// #include <algorithm>
// #include <cstdio>
// #include <vector>

// int get_max_capicity(const std::vector<int> &in_vec) {
//   if (in_vec.empty() || in_vec.size() < 2)
//     return 0;
//   int ret_value{0};
//   int left{0}, right{(int)in_vec.size() - 1};
//   while (left < right) {
//     int area = std::min(in_vec[left], in_vec[right]) * (right - left);
//     if (ret_value < area)
//       ret_value = area;
//     if (in_vec[left] < in_vec[right]) {
//       ++left;
//     } else {
//       --right;
//     }
//   }
//   // for (int i = 0; i < in_vec.size() - 1; ++i) {
//   //   for (int j = 1; j < in_vec.size(); ++j) {
//   //     int capacity = (j - i) * std::min(in_vec[i], in_vec[j]);
//   //     if (ret_value < capacity) {
//   //       ret_value = capacity;
//   //     }
//   //   }
//   // }
//   return ret_value;
// }

// int main() {
//   printf("%d\n", get_max_capicity(std::vector<int>{1, 8, 6, 2, 5, 4, 8, 3,
//   7})); printf("%d\n", get_max_capicity(std::vector<int>{1, 1})); return 0;
// }

//# 串联所有单词的子串
// #include <algorithm>
// #include <cstdio>
// #include <set>
// #include <string>
// #include <vector>

// std::vector<int> search_indexes(const std::string &in_str,
//                                 const std::vector<std::string> &words) {
//   std::vector<int> ret_vec;
//   if (in_str.empty() || words.empty())
//     return ret_vec;
//   int pos{0};
//   std::vector<int> poses;
//   poses.resize(words.size());
//   int status{1};
//   const int step{(int)(words[0].size() * (words.size() - 1))};
//   while (status != 0 && pos < in_str.size()) {
//     std::set<int> poses_set;
//     for (int i = 0; i < words.size(); ++i) {
//       poses[i] = in_str.find(words[i], pos);
//       if (poses[i] >= 0 && poses_set.count(poses[i]) > 0) {
//         poses[i] = in_str.find(words[i], poses[i] + 1);
//       }
//       if (poses[i] < 0) {
//         status = 0;
//         break;
//       }
//       poses_set.insert(poses[i]);
//     }
//     if (status != 0) {
//       std::sort(poses.begin(), poses.end(),
//                 [&](const auto &e1, const auto &e2) { return e1 < e2; });
//       if (*poses.rbegin() - *poses.begin() == step) {
//         ret_vec.emplace_back(*poses.begin());
//       }
//       pos = *poses.begin() + words[0].size();
//     }
//   }
//   return ret_vec;
// }

// int main() {
//   {
//     auto ret = search_indexes("barfoothefoobarman",
//                               std::vector<std::string>{"foo", "bar"});
//     std::for_each(ret.begin(), ret.end(),
//                   [](const auto &e) { printf("%d ", e); });
//     printf("\n");
//   }
//   {
//     auto ret = search_indexes(
//         "wordgoodgoodgoodbestwordgoodbestword",
//         std::vector<std::string>{"word", "good", "best", "word"});
//     std::for_each(ret.begin(), ret.end(),
//                   [](const auto &e) { printf("%d ", e); });
//     printf("\n");
//   }
//   return 0;
// }

//# 合并两个有序数组
// #include <algorithm>
// #include <cstdio>
// #include <cstring>
// #include <vector>

// void merge_array(std::vector<int> &out_vec, int size,
//                  const std::vector<int> &in_vec) {
//   if (in_vec.empty())
//     return;
//   int pos_out{0}, pos_in{0};
//   while (pos_out < size && pos_in < in_vec.size()) {
//     if (out_vec[pos_out] <= in_vec[pos_in]) {
//       ++pos_out;
//       continue;
//     }
//     memcpy(&out_vec[pos_out + 1], &out_vec[pos_out],
//            (size - pos_out) * sizeof(int));
//     out_vec[pos_out] = in_vec[pos_in];
//     ++pos_out;
//     ++pos_in;
//     ++size;
//   }
//   if (pos_in < in_vec.size()) {
//     memcpy(&out_vec[pos_out], &in_vec[pos_in],
//            (in_vec.size() - pos_in) * sizeof(int));
//   }
// }

// int main() {
//   std::vector<int> out{1, 2, 5, 6, 9, 10};
//   int size = out.size();
//   std::vector<int> in{3, 4, 7, 11};
//   out.resize(out.size() + in.size());
//   merge_array(out, size, in);
//   std::for_each(out.begin(), out.end(),
//                 [](const auto &e) { printf("%d ", e); });
//   printf("\n");
// }

//# 买卖股票的最佳时机
// #include <cstdio>
// #include <vector>

// int get_stock_profit(const std::vector<int> &in_vec) {
//   int ret_value{0};
//   for (int i = 0; i < in_vec.size() - 1; ++i) {
//     for (int j = i + 1; j < in_vec.size(); ++j) {
//       int profit = in_vec[j] - in_vec[i];
//       if (profit > ret_value) { ret_value = profit; }
//     }
//   }
//   return ret_value;
// }

// int get_stock_profit2(const std::vector<int> &in_vec, int start) {
//   int ret_value{0};
//   for (int i = start; i < in_vec.size() - 1; ++i) {
//     int max_profit{0};
//     for (int j = i + 1; j < in_vec.size(); ++j) {
//       int profit = in_vec[j] - in_vec[i];
//       if (profit > 0) { profit += get_stock_profit2(in_vec, j + 1); }
//       if (profit > max_profit) { max_profit = profit; }
//     }
//     if (max_profit > ret_value) { ret_value = max_profit; }
//   }
//   return ret_value;
// }

// int main() {
//   printf("%d\n", get_stock_profit(std::vector<int>{7, 1, 5, 3, 6, 4}));
//   printf("%d\n", get_stock_profit(std::vector<int>{1, 2, 3, 4, 5}));
//   printf("%d\n", get_stock_profit(std::vector<int>{7, 6, 4, 3, 1}));

//   printf("%d\n", get_stock_profit2(std::vector<int>{7, 1, 5, 3, 6, 4}, 0));
//   printf("%d\n", get_stock_profit2(std::vector<int>{1, 2, 3, 4, 5}, 0));
//   printf("%d\n", get_stock_profit2(std::vector<int>{7, 6, 4, 3, 1}, 0));
//   return 0;
// }

//# 长度最小的子数组
// #include <cstdio>
// #include <vector>

// using namespace std;

// int calc_sub(const vector<int> &in_vec, int target, int current, int count, int pos) {
//   int ret_value{0};
//   int i{pos};
//   while (i < in_vec.size()) {
//     if (in_vec[i] + current >= target) {
//       ret_value = count + 1;
//       break;
//     }
//     ++i;
//   }
//   if (i < in_vec.size()) {
//     for (int j = i + 1; j < in_vec.size(); ++j) {
//       int res = calc_sub(in_vec, target, 0, 0, j);
//       if (res > 0) {
//         if (ret_value == 0) {
//           ret_value = res;
//         } else if (res < ret_value) {
//           ret_value = res;
//         }
//       }
//     }
//   } else {
//     if (pos <= in_vec.size() - 1) { ret_value = calc_sub(in_vec, target, current + in_vec[pos], count + 1, pos + 1); }
//   }
//   return ret_value;
// }

// int main() {
//   printf("%d\n", calc_sub(vector<int>{2, 3, 1, 2, 4, 3}, 7, 0, 0, 0));
//   printf("%d\n", calc_sub(vector<int>{1, 4, 4}, 4, 0, 0, 0));
//   printf("%d\n", calc_sub(vector<int>{1, 1, 1, 1, 1, 1, 1, 1}, 11, 0, 0, 0));
//   return 0;
// }

//# 判断子序列
// bool check_sub(const string &in_str, const string &sub_str) {
//   bool ret_value{false};
//   int pos_in{0}, pos_sub{0};
//   while (pos_in < in_str.size()) {
//     if (in_str[pos_in] == sub_str[pos_sub]) {
//       ++pos_in;
//       ++pos_sub;
//       if (pos_sub >= sub_str.size()) {
//         ret_value = true;
//         break;
//       }
//     } else {
//       ++pos_in;
//     }
//   }
//   return ret_value;
// }

// int main() {
//   printf("%d\n", check_sub("ahbgdc", "abc"));
//   printf("%d\n", check_sub("ahbgdc", "axc"));
//   return 0;
// }

/*############################################################*/

#include <cstdio>
#include <vector>
#include <thread>
#include <algorithm>
#include <thread>
#include <mutex>
#include <deque>
#include <string>
#include <stack>
#include <map>
#include <set>

struct TreeNode {
  TreeNode* left{nullptr};
  TreeNode* right{nullptr};
  int value{0};
  int level{0};
};
void make_tree(std::vector<std::vector<std::string>> const& vv, TreeNode& t) {
  if (vv.empty()) { return; }
  int level{0};
  std::deque<TreeNode*> ndq;
  ndq.push_back(&t);
  t.value = std::atoi(vv[0][0].c_str());
  for (int i = 1; i < vv.size(); ++i) {
    // printf("level %d\n", level);
    int node_num = 2 * ++level;
    for (int j = 0; j < node_num && j < vv[i].size();) {
      TreeNode* left = new TreeNode;
      if (!vv[i][j].empty()) {
        left->value = std::atoi(vv[i][j++].c_str());
      } else {
        j++;
      }
      TreeNode* right = new TreeNode;
      if (!vv[i][j].empty()) {
        right->value = std::atoi(vv[i][j++].c_str());
      } else {
        j++;
      }
      left->level = right->level = level;
      TreeNode* parent = ndq.front();
      ndq.pop_front();
      parent->left = left;
      parent->right = right;
      ndq.push_back(left);
      ndq.push_back(right);
    }
  }
};
void in_traverse_tree(std::deque<TreeNode const*>& dq) {
  if (dq.empty()) return;
  static int level = 0;
  TreeNode const* pn = dq.front();
  dq.pop_front();
  if (pn != nullptr) {
    if (pn->level > level) {
      printf("\n");
      level = pn->level;
    }
    printf("%d ", pn->value);
    dq.push_back(pn->left);
    dq.push_back(pn->right);
  }
  in_traverse_tree(dq);
}
void print_tree(TreeNode const& t) {
  int level{0};
  std::deque<TreeNode const*> dq;
  dq.push_back(&t);
  in_traverse_tree(dq);
  printf("\n");
}

std::string get_max_common_prefix(std::vector<std::string> const& v) {
  std::string ret;
  if (v.empty()) return ret;
  int error = 0;
  int i = 1;
  for (; i <= v[0].size(); ++i) {
    ret = v[0].substr(0, i);
    for (int j = 1; j < v.size(); ++j) {
      if (v[j].find(ret) != 0) {
        error = 1;
        break;
      }
    }
    if (error != 0) { break; }
  }
  return ret.substr(0, i - 1);
}

std::string reverse_word(std::string const& s) {
  std::string rs{s.rbegin(), s.rend()};
  std::string ret;
  int i = 0, j = 0;
  for (; j < rs.size();) {
    if (rs[j] != ' ') {
      ++j;
      continue;
    }
    if (j > i) {
      std::string s{rs.rbegin() + rs.size() - j, rs.rbegin() + rs.size() - i};
      // printf("1 [%s]\n", s.c_str());
      ret += s;
      i = j;
    } else {
      while (rs[j] == ' ' && j < rs.size()) j++;
      std::string s{rs.rbegin() + rs.size() - j, rs.rbegin() + rs.size() - i};
      // printf("2 [%s]\n", s.c_str());
      ret += s;
      i = j;
    }
  }
  if (j > i) {
    std::string s{rs.rbegin() + rs.size() - j, rs.rbegin() + rs.size() - i};
    printf("3 [%s]\n", s.c_str());
    ret += s;
  }
  return ret;
}

int get_rain_capacity(std::vector<int> const& v) {
  int ret{0};
  if (v.empty()) return ret;
  int level{1};
  int stop{0};
  while (stop == 0) {
    stop = 1;
    int can_contain{0};
    int level_capacity{0};
    for (int i = 1; i < v.size(); ++i) {
      if (v[i - 1] > level) { stop = 0; }
      if (can_contain == 0) {
        if (v[i - 1] >= level && v[i] < level) {
          can_contain = 1;
          level_capacity = 1;
        }
      } else if (v[i] < level) {
        ++level_capacity;
      } else if (v[i] >= level) {
        can_contain = 0;
        ret += level_capacity;
        level_capacity = 0;
      }
    }
    ++level;
  }
  return ret;
}

std::string digital_roma(int v) {
  std::string ret;
  int d = v / 1000;
  if (d > 0) {
    for (int i = 0; i < d; ++i) ret += "M";
    v -= (d * 1000);
    d = v / 100;
  } else {
    d = v / 100;
  }
  if (d > 0) {
    switch (d) {
      case 1:
        ret += "C";
        break;
      case 2:
        ret += "CC";
        break;
      case 3:
        ret += "CCC";
        break;
      case 4:
        ret += "CD";
        break;
      case 5:
        ret += "D";
        break;
      case 6:
        ret += "DC";
        break;
      case 7:
        ret += "DCC";
        break;
      case 8:
        ret += "DCCC";
        break;
      case 9:
        ret += "CM";
        break;
    }
    v -= (d * 100);
    d = v / 10;
  } else {
    d = v / 10;
  }
  if (d > 0) {
    switch (d) {
      case 1:
        ret += "X";
        break;
      case 2:
        ret += "XX";
        break;
      case 3:
        ret += "XXX";
        break;
      case 4:
        ret += "XL";
        break;
      case 5:
        ret += "L";
        break;
      case 6:
        ret += "LX";
        break;
      case 7:
        ret += "LXX";
        break;
      case 8:
        ret += "LXXX";
        break;
      case 9:
        ret += "XC";
        break;
    }
    v -= (d * 10);
    d = v;
  }
  if (d > 0) {
    switch (d) {
      case 1:
        ret += "I";
        break;
      case 2:
        ret += "II";
        break;
      case 3:
        ret += "III";
        break;
      case 4:
        ret += "IV";
        break;
      case 5:
        ret += "V";
        break;
      case 6:
        ret += "VI";
        break;
      case 7:
        ret += "VII";
        break;
      case 8:
        ret += "VIII";
        break;
      case 9:
        ret += "IX";
        break;
    }
  }
  return ret;
}

int roma_digital(std::string const& s) {
  int ret{0};
  for (int i = 0; i < s.size();) {
    if (s[i] == 'M') {
      ret += 1000;
      ++i;
    } else if (s[i] == 'D') {
      ret += 500;
      ++i;
    } else if (s[i] == 'C') {
      if (i < s.size() - 1) {
        if (s[i + 1] == 'M') {
          ret += 900;
          i += 2;
        } else if (s[i + 1] == 'D') {
          ret += 400;
          i += 2;
        } else {
          ret += 100;
          ++i;
        }
      } else {
        ret += 100;
        ++i;
      }
    } else if (s[i] == 'L') {
      ret += 50;
      ++i;
    } else if (s[i] == 'X') {
      if (i < s.size() - 1) {
        if (s[i + 1] == 'C') {
          ret += 90;
          i += 2;
        } else if (s[i + 1] == 'L') {
          ret += 40;
          i += 2;
        } else {
          ret += 10;
          ++i;
        }
      } else {
        ret += 10;
        ++i;
      }
    } else if (s[i] == 'V') {
      ret += 5;
      ++i;
    } else if (s[i] == 'I') {
      if (i < s.size() - 1) {
        if (s[i + 1] == 'X') {
          ret += 9;
          i += 2;
        } else if (s[i + 1] == 'V') {
          ret += 4;
          i += 2;
        } else {
          ret += 1;
          ++i;
        }
      } else {
        ret += 1;
        ++i;
      }
    }
  }
  return ret;
}

void get_min_array(std::vector<int> const& v, int t, int& ret, int p = 0) {
  for (int q = p + 1; q < v.size(); ++q) {
    if (v[p] == t) {
      ret = 1;
      return;
    }
    if (v[p] + v[q] >= t) {
      if (ret == 0 || ret > 2) { ret = 2; }
    } else {
      int subret{0};
      get_min_array(v, t - v[p], subret, q);
      if (subret > 0) {
        if (ret == 0 || ret > subret + 1) { ret = subret + 1; }
      }
    }
  }
  if (p < v.size() - 1) {
    if (ret == 0 || ret > 1) { get_min_array(v, t, ret, p + 1); }
  }
}

bool is_sub_str(std::string const& s, std::string const& sub) {
  bool ret{false};
  if (s.empty() || sub.empty()) return ret;
  int subpos{0};
  int spos{0};
  int stop{0};
  for (; spos < s.size() && stop == 0;) {
    subpos = 0;
    for (int i = spos; i < s.size(); ++i) {
      if (s[i] == sub[subpos]) { ++subpos; }
      if (subpos >= sub.size()) {
        ret = true;
        stop = 1;
        break;
      }
    }
    if (stop == 0) { ++spos; }
  }
  return ret;
}

struct Element {
  int value{0};
  int op{0};
};
int calc(std::string const& exp) {
  int ret{0};
  if (exp.empty()) return ret;
  std::stack<std::string> st;
  std::vector<Element> vcsub;
  int number{0};
  for (int i = 0; i < exp.size(); ++i) {
    if (exp[i] == '(') {
      st.push("(");
    } else if (exp[i] == ')') {
      vcsub.clear();
      std::string subexp = st.top();
      while (subexp != "(" && !st.empty()) {
        if (subexp == "+") {
          vcsub.push_back({number, 1});
          st.pop();
        } else if (subexp == "-") {
          vcsub.push_back({number, 2});
          st.pop();
        } else {
          number = std::atoi(st.top().c_str());
          if (!vcsub.empty()) { vcsub.push_back({number, 0}); }
          st.pop();
        }
        if (st.empty()) break;
        subexp = st.top();
      }
      if (subexp != "(") {
        printf("bad expression\n");
        return 0;
      }
      st.pop();
      int subval{0};
      for (auto i = vcsub.rbegin(); i != vcsub.rend(); ++i) {
        if (i->op == 0) {
          subval = i->value;
        } else if (i->op == 1) {
          subval += i->value;
        } else if (i->op == 2) {
          subval -= i->value;
        }
      }
      st.push(std::to_string(subval));
    } else if (exp[i] == '+') {
      st.push("+");
    } else if (exp[i] == '-') {
      st.push("-");
    } else {
      if (st.top()[0] >= '0' && st.top()[0] <= '9') {
        st.top() += exp[i];
      } else {
        st.push("");
        st.top() += exp[i];
      }
    }
  }
  if (st.size() > 1) {
    vcsub.clear();
    std::string subexp = st.top();
    while (subexp != "(" && !st.empty()) {
      if (subexp == "+") {
        vcsub.push_back({number, 1});
        st.pop();
      } else if (subexp == "-") {
        vcsub.push_back({number, 2});
        st.pop();
      } else {
        number = std::atoi(st.top().c_str());
        if (!vcsub.empty()) { vcsub.push_back({number, 0}); }
        st.pop();
      }
      if (st.empty()) break;
      subexp = st.top();
    }
    int subval{0};
    for (auto i = vcsub.rbegin(); i != vcsub.rend(); ++i) {
      if (i->op == 0) {
        subval = i->value;
      } else if (i->op == 1) {
        subval += i->value;
      } else if (i->op == 2) {
        subval -= i->value;
      }
    }
    st.push(std::to_string(subval));
  }
  ret = std::atoi(st.top().c_str());
  return ret;
}

struct ListNode {
  ListNode* next{nullptr};
  int value{0};
};

void make_list(ListNode*& l, std::vector<int> const& v) {
  if (v.empty()) {
    l = nullptr;
    return;
  }
  l = new ListNode{nullptr, v[0]};
  ListNode* lp{l};
  for (int i = 1; i < v.size(); ++i) {
    ListNode* lc = new ListNode{nullptr, v[i]};
    lp->next = lc;
    lp = lc;
  }
}
void print_list(ListNode* l) {
  while (l != nullptr) {
    printf("%d ", l->value);
    l = l->next;
  }
  printf("\n");
}
void rotate_list(ListNode*& l, int s) {
  if (l == nullptr) return;
  if (s <= 0) return;
  ListNode* lo{l};
  ListNode* lp{nullptr};
  ListNode* ll{nullptr};
  for (int i = 0; i < s; ++i) {
    lp = lo;
    if (lo->next == nullptr) {
      ll = lo;
      lo = l;
    } else {
      lo = lo->next;
    }
  }
  if (lo == l) { return; }
  if (ll == nullptr) {
    for (ll = lo; ll->next != nullptr;) { ll = ll->next; }
  }
  lp->next = nullptr;
  ll->next = l;
  l = lo;
}
void reverse_list(ListNode*& l, int p0, int p1) {
  if (l == nullptr) return;
  if (p0 < 0 || p1 < 0 || p0 >= p1) return;
  ListNode* l0{l};
  ListNode* l1{l};
  ListNode* lp{nullptr};
  for (int i = 0; i < p0; ++i) {
    lp = l0;
    if (l0->next == nullptr)
      l0 = l;
    else
      l0 = l0->next;
  }
  for (int i = 0; i < p1; ++i) {
    if (l1->next == nullptr)
      l1 = l;
    else
      l1 = l1->next;
  }
  if (l1 <= l0) return;
  ListNode* next = l1->next;
  ListNode* lr0{nullptr};
  ListNode* lr1{nullptr};
  for (lr0 = l0, lr1 = l0->next; lr0 != l1;) {
    ListNode* l = lr1->next;
    lr1->next = lr0;
    lr0 = lr1;
    lr1 = l;
  }
  if (lp != nullptr) {
    l0->next = next;
    lp->next = l1;
  } else {
    l = l1;
    l0->next = next;
  }
}

int get_arrow_count(std::vector<std::vector<int>> const& vs) {
  if (vs.empty()) return 0;
  std::vector<std::vector<int>> svs;
  svs.emplace_back(vs[0]);
  for (int i = 1; i < vs.size(); ++i) {
    int merge = 0;
    for (int j = 0; j < svs.size();) {
      int xl0 = vs[i][0], xr0 = vs[i][1];
      int xl1 = svs[j][0], xr1 = svs[j][1];
      if (xr0 < xl1 || xr1 < xl0) {
        ++j;
      } else {
        int xl = xl0 < xl1 ? xl1 : xl0;
        int xr = xr0 < xr1 ? xr0 : xr1;
        svs[j][0] = xl;
        svs[j][1] = xr;
        merge = 1;
        break;
      }
    }
    if (merge == 0) { svs.emplace_back(vs[i]); }
  }
  return svs.size();
}

void merge_range(std::vector<std::vector<int>> const& vs, std::vector<std::vector<int>>& vr) {
  if (vs.empty()) {
    vr.clear();
    return;
  }
  vr.emplace_back(vs[0]);
  for (int i = 1; i < vs.size(); ++i) {
    int merge{0};
    for (int j = 0; j < vr.size(); ++j) {
      int xl0 = vs[i][0], xr0 = vs[i][1];
      int xl1 = vr[j][0], xr1 = vr[j][1];
      if (xr0 < xl1 || xl0 > xr1) { continue; }
      int xl = xl0 < xl1 ? xl0 : xl1;
      int xr = xr0 > xr1 ? xr0 : xr1;
      vr[j][0] = xl, vr[j][1] = xr;
      merge = 1;
      break;
    }
    if (merge == 0) { vr.emplace_back(vs[i]); }
  }
}

using range = std::vector<int>;
using ranges = std::vector<range>;
void insert_range(ranges& rs, range const& r) {
  if (rs.empty()) {
    rs.emplace_back(r);
    return;
  }
  for (int i = 0; i < rs.size();) {
    int xl = rs[i][0], xr = rs[i][1];
    if (r[1] < xl || r[0] > xr) {
      ++i;
      continue;
    }
    rs[i][0] = xl < r[0] ? xl : r[0], rs[i][1] = xr > r[1] ? xr : r[1];
    for (int j = i + 1; j < rs.size();) {
      xl = rs[i][0], xr = rs[i][1];
      if (rs[j][1] < xl || rs[j][0] > xr) { break; }
      rs[i][0] = xl < rs[j][0] ? xl : rs[j][0], rs[i][1] = xr > rs[j][1] ? xr : rs[j][1];
      rs.erase(rs.begin() + j);
    }
    break;
  }
}

int max_capacity(std::vector<int> const& v) {
  if (v.size() < 2) return 0;
  int ret{0};
  int stop{0};
  int left{0}, right{(int)v.size() - 1};
  while (stop == 0) {
    int val = (v[left] < v[right] ? v[left] : v[right]) * (right - left);
    if (val >= ret) {
      ret = val;
    } else {
      break;
    }
    if (v[left] < v[right] && left < right - 1 && v[left + 1] > v[left])
      ++left;
    else if (v[left] > v[right] && right > left + 1 && v[right - 1] > v[right])
      --right;
    else
      stop = 1;
  }
  return ret;
}

std::vector<int> get_substr_pos(std::string const& s, std::vector<std::string> const& vsub) {
  std::vector<int> vpos;
  if (s.empty() || vsub.empty()) return vpos;
  std::map<size_t, size_t> msubpos;
  int stop{0};
  int fpos{0};
  while (stop == 0 && fpos < s.size()) {
    msubpos.clear();
    for (int i = 0; i < vsub.size(); ++i) {
      size_t pos = s.find(vsub[i], fpos);
      if (pos == std::string::npos) {
        stop = 1;
        break;
      } else if (msubpos.count(pos) > 0) {
        pos = s.find(pos + msubpos[pos]);
        if (pos == std::string::npos) {
          stop = 1;
          break;
        }
        msubpos[pos] = vsub[i].size();
      } else {
        msubpos[pos] = vsub[i].size();
      }
    }
    if (stop == 0) {
      int unmatch{0};
      for (auto it = msubpos.begin(); it != msubpos.end();) {
        auto it0 = it;
        auto it1 = ++it;
        if (it1 == msubpos.end()) break;
        if (it0->first + it0->second != it1->first) {
          unmatch = 1;
          break;
        }
      }
      if (unmatch == 1) {
        fpos = msubpos.begin()->first + msubpos.begin()->second;
        continue;
      }
      vpos.emplace_back(msubpos.begin()->first);
      fpos = msubpos.begin()->first + msubpos.begin()->second;
    }
  }
  return vpos;
}

void merge_vector(std::vector<int>& ov, std::vector<int> const& iv) {
  if (iv.empty()) return;
  ov.resize(ov.size() + iv.size());
  int pos{0};
  int ovs{(int)ov.size() - (int)iv.size()};
  int i{0};
  for (; i < iv.size();) {
    if (ov[pos] > iv[i]) {
      std::copy(ov.begin() + pos, ov.begin() + pos + ovs, ov.begin() + pos + 1);
      ov[pos] = iv[i];
      ++i;
    } else {
      --ovs;
    }
    ++pos;
    if (ovs == 0) break;
  }
  if (i < iv.size()) { std::copy(iv.begin() + i, iv.end(), ov.begin() + pos); }
}

int main() {
  // std::vector<std::vector<std::string>> vv;
  // vv.emplace_back(std::vector<std::string>{"1"});
  // vv.emplace_back(std::vector<std::string>{"", "3"});
  // vv.emplace_back(std::vector<std::string>{"", "", "6", "7"});
  // TreeNode t;
  // make_tree(vv, t);
  // print_tree(t);

  // printf("%s\n", get_max_common_prefix(std::vector<std::string>{"atl", "atle", "atl"}).c_str());

  // printf("[%s]\n", reverse_word("   abc  def    ghi   ").c_str());

  // printf("%d\n", get_rain_capacity(std::vector<int>{4,2,0,3,2,5}));

  // printf("%s\n", digital_roma(3749).c_str());

  // printf("%d\n", roma_digital("MCMXCIV"));

  // int ret{0};
  // std::vector<int> v{1, 1, 1, 1, 1, 1, 1, 1};
  // get_min_array(v, 4, ret);
  // printf("%d\n", ret);

  // printf("%d\n", is_sub_str("ahbgdc", "abc"));

  // printf("%d\n", calc("(0)+(1+(1-3))"));

  // ListNode* l{nullptr};
  // make_list(l, std::vector<int>{1, 2, 3, 4, 5});
  // print_list(l);
  // rotate_list(l, 111);
  // print_list(l);
  // reverse_list(l, 36, 44);
  // print_list(l);

  // printf("%d\n", get_arrow_count(std::vector<std::vector<int>>{{10, 16}, {2, 8}, {1, 6}, {7, 12}}));
  // printf("%d\n", get_arrow_count(std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}}));
  // printf("%d\n", get_arrow_count(std::vector<std::vector<int>>{{1, 2}, {2, 3}, {3, 4}, {4, 5}}));

  // std::vector<std::vector<int>> res;
  // merge_range(decltype(res){{4, 7}, {1, 4}}, res);
  // for (auto const& e : res) { printf("[%d,%d]", e[0], e[1]); }
  // printf("\n");

  // ranges res{{{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}};
  // ranges res{{{1, 3}, {6, 9}}};
  // insert_range(res, range{2, 5});
  // for (auto const& e : res) { printf("[%d,%d]", e[0], e[1]); }
  // printf("\n");

  // printf("%d\n", max_capacity({1, 2, 3, 4, 5, 6, 7, 8, 9}));
  // printf("%d\n", max_capacity({1, 8, 6, 2, 5, 4, 8, 3, 7}));
  // printf("%d\n", max_capacity({1, 1}));

  // for (auto const& e : get_substr_pos("barfoofoobarthefoobarman", {"bar", "foo", "the"})) { printf("%d\n", e); }

  std::vector<int> ov{5, 6, 7};
  std::vector<int> iv{1, 2, 8};
  merge_vector(ov, iv);
  for (auto const& e : ov) { printf("%d ", e); }
  printf("\n");
  return 0;
}
