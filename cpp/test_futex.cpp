// 实现基于 futex 的信号量和互斥锁
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/time.h>
#include <climits>
#include <atomic>

class FutexMutex {
private:
    std::atomic<int> state{0}; // 0: unlocked, 1: locked

public:
    void lock() {
        int expected = 0;
        while (!state.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            if (expected == 1) {
                // 如果锁已被占用，则调用 futex 等待
                syscall(SYS_futex, &state, FUTEX_WAIT, 1, nullptr, nullptr, 0);
                expected = 0; // 重置 expected 值以重试
            }
        }
    }

    void unlock() {
        if (state.exchange(0, std::memory_order_release) == 1) {
            // 唤醒一个等待的线程
            syscall(SYS_futex, &state, FUTEX_WAKE, 1, nullptr, nullptr, 0);
        }
    }
};

using namespace std;

// 测试函数
void test_mutex(FutexMutex& mutex, int& counter, int id) {
    for (int i = 0; i < 100000; ++i) {
        mutex.lock();
        counter++;
        mutex.unlock();
    }
    cout << "Thread " << id << " finished" << endl;
}

int main() {
    FutexMutex mutex;
    int counter = 0;
    
    thread t1(test_mutex, std::ref(mutex), std::ref(counter), 1);
    thread t2(test_mutex, std::ref(mutex), std::ref(counter), 2);
    
    t1.join();
    t2.join();
    
    cout << "Counter value: " << counter << endl;
    
    return 0;
}
