#pragma once
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>

template<typename bufferType, typename extraDataType, int capacity, int buffer_size, bool wait_on_empty>
class BufferProcessQueue {
public:
    typedef bufferType BufferType;
    struct BufferBlock {
        typedef extraDataType ExtraDataType;
        ExtraDataType *pExtraData = nullptr;
        BufferType *pBuffer = nullptr;
        int real_size = 0;
    };
    static const int capacity_val{capacity};
    static const int buffer_size_val{buffer_size};
    static const bool wait_on_empty_val{wait_on_empty};

    BufferProcessQueue(const std::function<void(extraDataType *pExtraData)> *init) {
        const int extraDataTypeSize = sizeof(extraDataType);
        for (auto &buf : buf_array) {
            buf.resize(extraDataTypeSize + buffer_size);
            static int idx = 0;
            m_buf_blk_array[idx].pExtraData = (extraDataType *)&buf[0];
            m_buf_blk_array[idx].pBuffer = (bufferType *)m_buf_blk_array[idx].pExtraData + extraDataTypeSize;
            m_buf_blk_array[idx].real_size = buf.size() - extraDataTypeSize;
            if (init && *init) {
                (*init)(m_buf_blk_array[idx].pExtraData);
            }
            buffer_queue.push_back(&m_buf_blk_array[idx]);
            idx++;
        }
    }

    ~BufferProcessQueue() {
    }

    int produce(const std::function<void(BufferBlock *)> &call_back) {
        BufferBlock *pBufBlk = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_qmutex);
            if (buffer_queue.empty()) {
                if(process_queue.empty()) {
                    printf("producer need to wait\n");
                    if (wait_on_empty) {
                        m_qcv.wait(lock, [this] { return !buffer_queue.empty() || !process_queue.empty(); });
                    }
                    return -1;
                }
                pBufBlk = process_queue.front();
                process_queue.pop_front();
            } else {
                pBufBlk = buffer_queue.front();
                buffer_queue.pop_front();
            }
        }
        call_back(pBufBlk);
        {
            std::unique_lock<std::mutex> lock(m_qmutex);
            process_queue.push_back(pBufBlk);
        }
        if (wait_on_empty) {
            m_qcv.notify_one();
        }
        // {
        //     std::unique_lock<std::mutex> lock(m_qmutex);
        //     printf("buffer queue size %d\n", buffer_queue.size());
        //     printf("process queue size %d\n", process_queue.size());
        // }
        return 0;
    }

    int consume(const std::function<void(BufferBlock *)> &call_back) {
        BufferBlock *pBufBlk = nullptr;
        bool empty = true;
        {
            std::unique_lock<std::mutex> lock(m_qmutex);
            if (!process_queue.empty()) {
                pBufBlk = process_queue.front();
                process_queue.pop_front();
                empty = false;
            }
        }
        if (!empty) {
            call_back(pBufBlk);
            {
                std::unique_lock<std::mutex> lock(m_qmutex);
                buffer_queue.push_back(pBufBlk);
                if (wait_on_empty) {
                    m_qcv.notify_one();
                }
            }
        } else {
            if (wait_on_empty) {
                std::unique_lock<std::mutex> lock(m_qmutex);
                m_qcv.wait(lock, [this] { return !process_queue.empty(); });
            }
            return -1;
        }
        // {
        //     std::unique_lock<std::mutex> lock(m_qmutex);
        //     printf("buffer queue size %d\n", buffer_queue.size());
        //     printf("process queue size %d\n", process_queue.size());
        // }
        return 0;
    }

private:
    std::deque<BufferBlock *>process_queue;
    std::deque<BufferBlock *>buffer_queue;
    std::mutex m_qmutex;
    std::vector<BufferType> buf_array[capacity];
    std::condition_variable m_qcv;
    BufferBlock m_buf_blk_array[capacity];
};
