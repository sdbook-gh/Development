#ifndef UPLOAD_VEHICLE_REPORT_INCLUDE_LOCKFREEQUEUE_H
#define UPLOAD_VEHICLE_REPORT_INCLUDE_LOCKFREEQUEUE_H

#include <atomic>
#include <memory>

#define PRINT_INFO printf
#define PRINT_ERROR printf

namespace v2x {
    template<typename T, size_t chunk_size, typename allocator_t = std::allocator<T>>
    class LockFreeQueue {
        static_assert(chunk_size > 4, "chunk_size must > 4");

    private:
        struct Chunk {
            Chunk *pNext = nullptr;
            T *pValueArray = nullptr;
        };
        using chunk_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<Chunk>;
        allocator_t allocator;
        chunk_allocator_t chunk_allocator;

    public:
        explicit LockFreeQueue() {
            pFront = alloc_chunk();
            if (pFront == nullptr) {
                PRINT_ERROR("pushTail err: alloc init");
            }
            pTail.store(pFront);
            valid = true;
            segment = 1;
        }

        inline bool
        empty() const {
            return queue_len == 0;
        }

        inline int
        size() const {
            return queue_len;
        }

        bool
        pushHead(const T &value) {
            if (!valid) {
                PRINT_ERROR("pushHead err: init");
                return false;
            }
            Chunk *lockFront = nullptr;
            Chunk *lockTail = nullptr;
            bool acquired = false;
            while (true) {
                if (lockFront == nullptr) {
                    lockFront = pFront;
                    continue;
                }
                if (lockTail == nullptr) {
                    lockTail = pTail;
                    continue;
                }
                if (pFront.compare_exchange_weak(lockFront, nullptr)) {
                    std::shared_ptr<int> guardFront(nullptr, [this, &lockFront, &acquired](void *) {
                        if (!acquired)
                            pFront = lockFront;
                    });
                    if (pTail.compare_exchange_weak(lockTail, nullptr)) {
                        acquired = true;
                        break;
                    }
                }
            }
            std::shared_ptr<int> guardFront(nullptr, [this, &lockFront](void *) { pFront = lockFront; });
            std::shared_ptr<int> guardTail(nullptr, [this, &lockTail](void *) { pTail = lockTail; });
            if (front_pos > 0) {
                lockFront->pValueArray[--front_pos] = value;
                front_size++;
                queue_len++;
                return true;
            }
            if (front_size < chunk_size) {
                for (int pos = front_size; pos > 0; pos--) {
                    lockFront->pValueArray[pos] = lockFront->pValueArray[pos - 1];
                }
                lockFront->pValueArray[front_pos] = value;
                if (lockFront == lockTail) {
                    tail_pos++;
                }
                front_size++;
                queue_len++;
                return true;
            }
            Chunk *pChunk = alloc_chunk();
            if (pChunk == nullptr) {
                PRINT_ERROR("pushHead err: alloc head");
                return false;
            }
            segment++;
            pChunk->pNext = lockFront;
            lockFront = pChunk;
            front_pos = front_size = 0;
            lockFront->pValueArray[front_pos++] = value;
            front_size++;
            queue_len++;
            return true;
        }

        bool
        pushTail(const T &value) {
            if (!valid) {
                PRINT_ERROR("pushTail err: init");
                return false;
            }
            Chunk *lock = nullptr;
            while (true) {
                if (lock == nullptr) {
                    lock = pTail;
                    continue;
                }
                if (pTail.compare_exchange_weak(lock, nullptr)) {
                    break;
                }
            }
            std::shared_ptr<int> guard(nullptr, [this, &lock](void *) { pTail = lock; });
            if (tail_pos >= chunk_size) {
                Chunk *pChunk = alloc_chunk();
                if (pChunk == nullptr) {
                    PRINT_ERROR("pushTail err: alloc tail");
                    return false;
                }
                segment++;
                lock->pNext = pChunk;
                lock = pChunk;
                tail_pos = 0;
            }
            lock->pValueArray[tail_pos++] = value;
            queue_len++;
            return true;
        }

        bool
        popHead(T &value) {
            if (!valid) {
                PRINT_ERROR("popHead err: init");
                return false;
            }
            Chunk *lock = nullptr;
            while (true) {
                if (lock == nullptr) {
                    lock = pFront;
                    continue;
                }
                if (pFront.compare_exchange_weak(lock, nullptr)) {
                    break;
                }
            }
            std::shared_ptr<int> guard(nullptr, [this, &lock](void *) { pFront = lock; });
            if (queue_len == 0)
            {
                return false;
            }
            if (front_pos >= chunk_size) {
                if (lock->pNext != nullptr) {
                    Chunk *pChunk = lock;
                    lock = lock->pNext;
                    dealloc_chunk(pChunk);
                    front_pos = 0;
                    front_size = chunk_size;
                } else {
                    return false;
                }
            }
            value = lock->pValueArray[front_pos++];
            front_size--;
            queue_len--;
            return true;
        }

        bool
        popTail(T &value) {
            if (!valid) {
                PRINT_ERROR("popTail err: init");
                return false;
            }
            Chunk *lockFront = nullptr;
            Chunk *lockTail = nullptr;
            bool acquired = false;
            while (true) {
                if (lockFront == nullptr) {
                    lockFront = pFront;
                    continue;
                }
                if (lockTail == nullptr) {
                    lockTail = pTail;
                    continue;
                }
                if (pFront.compare_exchange_weak(lockFront, nullptr)) {
                    std::shared_ptr<int> guardFront(nullptr, [this, &lockFront, &acquired](void *) {
                        if (!acquired)
                            pFront = lockFront;
                    });
                    if (pTail.compare_exchange_weak(lockTail, nullptr)) {
                        acquired = true;
                        break;
                    }
                }
            }
            std::shared_ptr<int> guardFront(nullptr, [this, &lockFront](void *) { pFront = lockFront; });
            std::shared_ptr<int> guardTail(nullptr, [this, &lockTail](void *) { pTail = lockTail; });
            if (queue_len == 0) {
                return false;
            }
            if (tail_pos > 0) {
                value = lockTail->pValueArray[--tail_pos];
                queue_len--;
                return true;
            }
            Chunk *pPrevTail = lockFront;
            while (pPrevTail->pNext != lockTail && pPrevTail->pNext != nullptr) {
                pPrevTail = pPrevTail->pNext;
            }
            if (pPrevTail->pNext != nullptr) {
                dealloc_chunk(lockTail);
                segment--;
                int currentsegment = segment;
            }
            lockTail = pPrevTail;
            lockTail->pNext = nullptr;
            tail_pos = chunk_size;
            value = lockTail->pValueArray[--tail_pos];
            queue_len--;
            return true;
        }

    private:
        bool valid{false};
        std::atomic<Chunk *> pFront{nullptr};
        volatile size_t front_pos{0};
        volatile size_t front_size{0};
        std::atomic<Chunk *> pTail{nullptr};
        volatile size_t tail_pos{0};
        volatile int queue_len{0};
        volatile int segment{0};

        Chunk *
        alloc_chunk() {
            Chunk *pChunk = chunk_allocator.allocate(1);
            if (pChunk != nullptr) {
                pChunk->pValueArray = allocator.allocate(chunk_size);
                if (pChunk->pValueArray != nullptr) {
                    pChunk->pNext = nullptr;
                } else {
                    chunk_allocator.deallocate(pChunk, 1);
                    pChunk = nullptr;
                }
            }
            return pChunk;
        }

        void
        dealloc_chunk(Chunk *pChunk) {
            if (pChunk != nullptr) {
                if (pChunk->pValueArray != nullptr) {
                    allocator.deallocate(pChunk->pValueArray, chunk_size);
                }
                chunk_allocator.deallocate(pChunk, 1);
            }
        }

        void
        dump() {
            if (!valid) {
                PRINT_ERROR("dump err: init");
                return;
            }
            Chunk *lockFront = nullptr;
            while (true) {
                if (lockFront == nullptr) {
                    lockFront = pFront;
                    continue;
                }
                if (pFront.compare_exchange_weak(lockFront, nullptr)) {
                    break;
                }
            }
            std::shared_ptr<int> guardFront(nullptr, [this, &lockFront](void *) { pFront = lockFront; });
            Chunk *pPrevTail = lockFront;
            while (pPrevTail->pNext != nullptr) {
//			printf("%lx\n", pPrevTail->pValueArray);
                pPrevTail = pPrevTail->pNext;
            }
        }
    };
}
#endif //UPLOAD_VEHICLE_REPORT_INCLUDE_LOCKFREEQUEUE_H
