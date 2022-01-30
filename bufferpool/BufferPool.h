#pragma once
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <cassert>

template<typename bufferType, typename extraDataType, int capacity, int buffer_size, bool wait_on_empty>
class BufferPool {
public:
    static const int capacity_val{capacity};
    static const int buffer_size_val{buffer_size};
    static const bool wait_on_empty_val{wait_on_empty};
    static const int extraDataTypeSize{sizeof(extraDataType)};

    typedef bufferType BufferType;
    struct BufferBlock {
        typedef extraDataType ExtraDataType;
        ExtraDataType *pExtraData{nullptr};
        static const int ExtraDataTypeSize{sizeof(extraDataType)};
        BufferType *pBuffer{nullptr};
        int real_size{0};
        static const int BufferTypeSize{sizeof(BufferType)};
        bool need_print{false};
        BufferBlock() {
            if (need_print)
                printf("BufferBlock\n");
        }
        BufferBlock(const BufferBlock &other) {
            if (need_print)
                printf("BufferBlock(const BufferBlock &other)\n");
            pExtraData = other.pExtraData;
            pBuffer = other.pBuffer;
            real_size = other.real_size;
            pool = other.pool;
        }
        ~BufferBlock() {
            if (need_print)
                printf("~BufferBlock\n");
            if (pool != nullptr) {
                pool->release(*this);
                if (need_print)
                    printf("++++++++++++++++ release buffer\n");
            }
        }
    private:
        mutable BufferPool<BufferType, extraDataType, capacity, buffer_size, wait_on_empty> *pool{nullptr};
        friend class BufferPool<BufferType, extraDataType, capacity, buffer_size, wait_on_empty>;
    };

    BufferPool(const std::function<void(extraDataType *pExtraData)> *init) {
        const int extraDataTypeSize = sizeof(extraDataType);
        for (auto &buf : buf_array) {
            buf.resize(extraDataTypeSize + buffer_size * sizeof(BufferType));
            BufferBlock bufferBlock;
            bufferBlock.pExtraData = (extraDataType *) &buf[0];
            bufferBlock.pBuffer = (BufferType *) (&buf[0] + extraDataTypeSize);
            bufferBlock.real_size = buffer_size;
            if (init && *init) {
                (*init)(bufferBlock.pExtraData);
            }
            buffer_queue.push_back(bufferBlock);
        }
    }

    ~BufferPool() {
    }

    int acquire(BufferBlock &bufBlk) {
        static int acquire_count = 0;
        {
            std::unique_lock<std::mutex> lock(m_qmutex);
            acquire_count++;
            if (bufBlk.need_print)
                printf("acquire buffer count %d queue size %d\n", acquire_count, buffer_queue.size());
            if (buffer_queue.empty()) {
                if (bufBlk.need_print)
                    printf("acquire buffer queue need to wait++++++++++++++++\n");
                if (!wait_on_empty) {
                    return -1;
                }
                m_qcv.wait(lock, [this] { return !buffer_queue.empty(); });
            }
            bufBlk = buffer_queue.front();
            bufBlk.pool = this;
            buffer_queue.pop_front();
        }
        // {
        //     std::unique_lock<std::mutex> lock(m_qmutex);
        //     printf("acquire buffer queue size %d\n", buffer_queue.size());
        // }
        return 0;
    }

    int release(const BufferBlock &bufBlk) {
        static int release_count = 0;
        {
            std::unique_lock<std::mutex> lock(m_qmutex);
            release_count++;
            if (bufBlk.need_print)
                printf("release buffer count %d queue size %d\n", release_count, buffer_queue.size());
            BufferBlock newBufBlk;
            newBufBlk.pExtraData = bufBlk.pExtraData;
            newBufBlk.pBuffer = bufBlk.pBuffer;
            newBufBlk.real_size = buffer_size;
            bufBlk.pool = nullptr;
            buffer_queue.push_back(newBufBlk);
            if (wait_on_empty) {
                m_qcv.notify_one();
            }
        }
        // {
        //     std::unique_lock<std::mutex> lock(m_qmutex);
        //     printf("release buffer queue size %d\n", buffer_queue.size());
        // }
        return 0;
    }

private:
    std::deque<BufferBlock> buffer_queue;
    std::mutex m_qmutex;
    std::vector<uint8_t> buf_array[capacity];
    std::condition_variable m_qcv;
};
