#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iterator>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "LockFreeQueue.h"
#include <array>
#include <boost/thread/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/atomic.hpp>
#include "concurrentqueue.h"

struct Test {
    std::array<int, 1000> value;

    Test() = default;
};

int
main() {
    using namespace v2x;
#if 1
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    LockFreeQueue<Test, 64> queue;
    std::atomic_bool b_push_finished = {false};
    std::thread xthread_in([&queue, &b_push_finished, &start](void) -> void {
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000000; ++i) {
            Test test;
            queue.pushTail(test);
            queue.pushHead(test);
        }

        b_push_finished = true;
    });

    std::thread xthread_out([&queue, &b_push_finished, &end](void) -> void {
        while (true) {
//            while (b_push_finished == false) {
//                std::this_thread::sleep_for(std::chrono::seconds(1));
//            }
            if (!queue.empty()) {
//                printf("size:%d\n", queue.size());
                Test value;
                queue.popHead(value);
                queue.popTail(value);
//										std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (b_push_finished && queue.empty()) {
                end = std::chrono::steady_clock::now();
                break;
            }
        }
    });

    std::thread xthread_out2([&queue, &b_push_finished, &end](void) -> void {
        while (true) {
//            while (b_push_finished == false) {
//                std::this_thread::sleep_for(std::chrono::seconds(1));
//            }
            if (!queue.empty()) {
//                printf("size:%d\n", queue.size());
                Test value;
                queue.popHead(value);
                queue.popTail(value);
//										std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (b_push_finished && queue.empty()) {
                end = std::chrono::steady_clock::now();
                break;
            }
        }
    });

    xthread_in.join();
    xthread_out.join();
    xthread_out2.join();
#endif

#if 0
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    boost::lockfree::queue<Test, boost::lockfree::capacity<1024> > queue;
    std::atomic_bool b_push_finished = {false};
    std::thread xthread_in([&queue, &b_push_finished, &start](void) -> void {
        start = std::chrono::steady_clock::now();
        for (int i = 1; i < 100000; ++i) {
            Test test;
            queue.push(test);
        }
        b_push_finished = true;
    });

    std::thread xthread_out([&queue, &b_push_finished, &end](void) -> void {
        while (true) {
            if (!queue.empty()) {
                Test value;
                queue.pop(value);
            }
            if (b_push_finished && queue.empty()) {
                end = std::chrono::steady_clock::now();
                break;
            }
        }
    });
    std::thread xthread_out2([&queue, &b_push_finished, &end](void) -> void {
        while (true) {
            if (!queue.empty()) {
                Test value;
                queue.pop(value);
            }
            if (b_push_finished && queue.empty()) {
                end = std::chrono::steady_clock::now();
                break;
            }
        }
    });
    xthread_in.join();
    xthread_out.join();
    xthread_out2.join();
#endif

#if 0
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    moodycamel::ConcurrentQueue<Test> queue;
    std::atomic_bool b_push_finished = {false};
    std::thread xthread_in([&queue, &b_push_finished, &start](void) -> void {
        start = std::chrono::steady_clock::now();
        for (int i = 1; i < 100000; ++i) {
            Test test;
            queue.enqueue(test);
        }
        b_push_finished = true;
    });

    std::thread xthread_out([&queue, &b_push_finished, &end](void) -> void {
        while (true) {
            if (queue.size_approx() > 0) {
                Test value;
                queue.try_dequeue(value);
            }
            if (b_push_finished && queue.size_approx() == 0) {
                end = std::chrono::steady_clock::now();
                break;
            }
        }
    });
    std::thread xthread_out2([&queue, &b_push_finished, &end](void) -> void {
        while (true) {
            if (!queue.size_approx() > 0) {
                Test value;
                queue.try_dequeue(value);
            }
            if (b_push_finished && queue.size_approx() == 0) {
                end = std::chrono::steady_clock::now();
                break;
            }
        }
    });
    xthread_in.join();
    xthread_out.join();
    xthread_out2.join();
#endif

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    return 0;
}
