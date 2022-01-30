#pragma once

#include <string>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <fcntl.h>      /* For O_CREAT, O_RDWR */
#include <sys/mman.h>   /* shared memory and mmap() */
#include <sys/stat.h>   /* S_IRWXU */

static void *get_shared_memory(const std::string &shm_name, bool master, int shm_size) {
    int shm_fd = 0;
    if (master) {
        shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRWXU | S_IRWXG);
    } else {
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, S_IRWXU | S_IRWXG);
    }
    if (shm_fd < 0) {
        perror("shm_open failed");
        return nullptr;
    }
    if (master) {
        if (ftruncate(shm_fd, shm_size) != 0) {
            perror("ftruncate failed");
        }
    }
    void *shm_header = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_header == MAP_FAILED || shm_header == nullptr) {
        perror("mmap failed");
        return nullptr;
    }
    return shm_header;
}

class ShmMutex {
public:
    pthread_mutex_t _mutex;
    pthread_mutexattr_t _attr;

    void init(bool pshared = false) {
        pthread_mutexattr_init(&_attr);
        if (pshared)
            pthread_mutexattr_setpshared(&_attr, PTHREAD_PROCESS_SHARED);
        else
            pthread_mutexattr_setpshared(&_attr, PTHREAD_PROCESS_PRIVATE);
        pthread_mutex_init(&_mutex, &_attr);
    }

    int lock() {
        return pthread_mutex_lock(&_mutex);
    }

    int trylock() {
        return pthread_mutex_trylock(&_mutex);
    }

    int unlock() {
        return pthread_mutex_unlock(&_mutex);
    }
};

class ShmCond {
public:
    pthread_cond_t _cond;
    pthread_condattr_t _attr;

    void init(bool pshared = false) {
        pthread_condattr_init(&_attr);
        if (pshared)
            pthread_condattr_setpshared(&_attr, PTHREAD_PROCESS_SHARED);
        else
            pthread_condattr_setpshared(&_attr, PTHREAD_PROCESS_PRIVATE);
        pthread_cond_init(&_cond, &_attr);
    }

    int wait(ShmMutex &m) {
        return pthread_cond_wait(&_cond, &m._mutex);
    }

    int timedwait(const struct timespec &ts, ShmMutex &m) {
        return pthread_cond_timedwait(&_cond, &m._mutex, &ts);
    }

    int signal() {
        return pthread_cond_signal(&_cond);
    }

    int broadcast() {
        return pthread_cond_broadcast(&_cond);
    }
};

template<typename T>
class ShmRingBufferQueue {
public:
    ShmRingBufferQueue(const std::string &inShmName, bool inmaster, int incap) :
            buffer_capacity(incap), shm_name(inShmName), master(inmaster) {
    }

    ~ShmRingBufferQueue() {
        if (shm_header) {
            munmap((void *) shm_header, shm_size);
        }
        shm_header = nullptr;
        buffer_header = nullptr;
    }

    bool init() {
        shm_size = sizeof(ShmHeader) + buffer_capacity * sizeof(T);
        shm_header = (ShmHeader *) get_shared_memory(shm_name, master, shm_size);
        if (shm_header == nullptr) {
            printf("shared memory alloc failed");
            return false;
        }
        buffer_header = (T *) ((uint8_t *) shm_header + sizeof(ShmHeader));
        if (master) {
            shm_header->_capacity = buffer_capacity;
            shm_header->_begin = shm_header->_end = 0;
            shm_header->_size = 0;
        }
        printf("shm_header->_capacity:%d\n", shm_header->_capacity);
        printf("shm_header->_begin:%d\n", shm_header->_begin);
        printf("shm_header->_end:%d\n", shm_header->_end);
        printf("shm_header->_size:%d\n", shm_header->_size);
        return true;
    }

    int capacity() const { return shm_header->_capacity; };

    void clear() {
        if (shm_header == nullptr) {
            printf("ShmRingBufferQueue init failed\n");
            return;
        }
        shm_header->_begin = shm_header->_end = shm_header->_size = 0;
    }

    bool push_back(const T &e) {
        if (shm_header == nullptr) {
            printf("ShmRingBufferQueue init failed\n");
            return false;
        }
        if (shm_header->_size >= shm_header->_capacity) {
            return false;
        }
        buffer_header[shm_header->_end] = e;
        shm_header->_end = (shm_header->_end + 1) % shm_header->_capacity;
        ++shm_header->_size;
        //        printf("push_back\n");
        return true;
    }

    bool pop_front(T &e) {
        if (shm_header == nullptr) {
            printf("ShmRingBufferQueue init failed\n");
            return false;
        }
        if (shm_header->_size <= 0) {
            return false;
        }
        e = buffer_header[shm_header->_begin];
        shm_header->_begin = (shm_header->_begin + 1) % shm_header->_capacity;
        --shm_header->_size;
        //        printf("pop_front\n");
        return true;
    }

    int size() {
        return shm_header->_size;
    }

    bool empty() {
        return shm_header->_size == 0;
    }

    void resize(int size) {
        shm_header->_size = size > shm_header->_capacity ? shm_header->_capacity : size;
        shm_header->_end = (shm_header->_begin + shm_header->_size) % shm_header->_capacity;
    }

private:

    struct ShmHeader {
        int _capacity{0};
        int _begin{0};
        int _end{0};
        int _size{0};
    };

    ShmHeader *shm_header{nullptr};
    T *buffer_header{nullptr};
    int buffer_capacity{0};
    std::string shm_name;
    int shm_size{0};
    bool master{false};
};

#include <functional>

template<typename bufferType, int bufferBlockSize, int capacity, bool is_master, bool wait_on_empty>
class ShmBufferProcessQueue {
public:
    typedef bufferType BufferType;
    struct BufferBlock {
        BufferType buffer[bufferBlockSize];
        int real_size{0};
    };
    static const int capacity_val{capacity};
    static const bool wait_on_empty_val{wait_on_empty};

    ShmBufferProcessQueue(const std::string &qName)
            : shm_name(qName),
              process_queue(qName + "_pq", is_master, capacity),
              buffer_queue(qName + "bq", is_master, capacity) {
    }

    ~ShmBufferProcessQueue() {
    }

    bool init() {
        int shm_size = sizeof(ShmMutex) + sizeof(ShmCond) + sizeof(BufferBlock) * capacity;
        void *basePtr = get_shared_memory(shm_name, is_master, shm_size);
        if (basePtr == nullptr) {
            perror("ShmBufferProcessQueue shared memory alloc failed\n");
            return false;
        }
        m_qmutex = (ShmMutex *) basePtr;
        m_qcv = (ShmCond *) ((uint8_t *) m_qmutex + sizeof(ShmMutex));
        m_buf_blk_array = (BufferBlock *) ((uint8_t *) m_qcv + sizeof(ShmCond));
        printf("ShmBufferProcessQueue m_qmutex:%x\n", m_qmutex);
        printf("ShmBufferProcessQueue m_qcv:%x\n", m_qcv);
        printf("ShmBufferProcessQueue m_buf_blk_array:%x\n", m_buf_blk_array);
        if (is_master) {
            m_qmutex->init(is_master);
            m_qcv->init(is_master);
        }
        if (!process_queue.init()) {
            perror("ShmBufferProcessQueue process_queue init failed\n");
            return false;
        }
        if (!buffer_queue.init()) {
            printf("ShmBufferProcessQueue buffer_queue init failed\n");
            return false;
        }
        if (is_master) {
            for (int i = 0; i < capacity; ++i) {
                m_buf_blk_array[i].buffer[0] = i;
                m_buf_blk_array[i].real_size = 0;
                printf("%d, %d\n", m_buf_blk_array[i].buffer[0], m_buf_blk_array[i].real_size);
                buffer_queue.push_back(i);
            }
        } else {
            for (int i = 0; i < capacity; ++i) {
                printf("%d, %d\n", m_buf_blk_array[i].buffer[0], m_buf_blk_array[i].real_size);
            }
        }
        return true;
    }

    int produce(const std::function<void(BufferBlock *)> &call_back) {
        int bufBlkIdx;
        {
            m_qmutex->lock();
            if (buffer_queue.empty()) {
                if (process_queue.empty()) {
                    printf("producer need to wait\n");
                    if (wait_on_empty) {
                        do {
                            m_qcv->wait(*m_qmutex);
                        } while (buffer_queue.empty() && process_queue.empty());
                    }
                    m_qmutex->unlock();
                    return -1;
                }
                process_queue.pop_front(bufBlkIdx);
            } else {
                buffer_queue.pop_front(bufBlkIdx);
            }
            m_qmutex->unlock();
        }
        call_back(m_buf_blk_array + bufBlkIdx);
        {
            m_qmutex->lock();
            process_queue.push_back(bufBlkIdx);
            m_qmutex->unlock();
        }
        if (wait_on_empty) {
            m_qcv->signal();
        }
        {
            m_qmutex->lock();
            printf("produce buffer queue size %d\n", buffer_queue.size());
            printf("produce process queue size %d\n", process_queue.size());
            m_qmutex->unlock();
        }
        return 0;
    }

    int consume(const std::function<void(BufferBlock *)> &call_back) {
        int bufBlkIdx;
        bool empty = true;
        {
            m_qmutex->lock();
            if (!process_queue.empty()) {
                process_queue.pop_front(bufBlkIdx);
                empty = false;
            }
            m_qmutex->unlock();
        }
        if (!empty) {
            call_back(m_buf_blk_array + bufBlkIdx);
            {
                m_qmutex->lock();
                buffer_queue.push_back(bufBlkIdx);
                if (wait_on_empty) {
                    m_qcv->signal();
                }
                m_qmutex->unlock();
            }
        } else {
            if (wait_on_empty) {
                m_qmutex->lock();
                do {
                    m_qcv->wait(*m_qmutex);
                } while (process_queue.empty());
                m_qmutex->unlock();
            }
            return -1;
        }
        {
            m_qmutex->lock();
            printf("consume buffer queue size %d\n", buffer_queue.size());
            printf("consume process queue size %d\n", process_queue.size());
            m_qmutex->unlock();
        }
        return 0;
    }

private:
    ShmRingBufferQueue<int> process_queue;
    ShmRingBufferQueue<int> buffer_queue;
    std::string shm_name;
    ShmMutex *m_qmutex{nullptr};
    ShmCond *m_qcv{nullptr};
    BufferBlock *m_buf_blk_array{nullptr};
};
