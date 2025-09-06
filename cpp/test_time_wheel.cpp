#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

// 时间轮类
class TimeWheel {
public:
    // 任务结构
typedef std::function<void()> Task;

private:
    // 时间轮的槽数
    size_t slots_;
    // 当前指向的槽
    size_t current_slot_;
    // 每个槽的时间间隔（毫秒）
    size_t interval_;
    // 时间轮的槽
    std::vector<std::vector<Task>> wheel_;
    // 互斥锁
    std::mutex mutex_;
    // 条件变量
    std::condition_variable cv_;
    // 是否停止
    bool stop_;
    // 轮询线程
    std::thread worker_;

    // 轮询函数
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(interval_), [this] { return stop_; });
            
            if (stop_) {
                break;
            }
            
            // 执行当前槽中的所有任务
            for (auto& task : wheel_[current_slot_]) {
                task();
            }
            
            // 清空当前槽
            wheel_[current_slot_].clear();
            
            // 移动到下一个槽
            current_slot_ = (current_slot_ + 1) % slots_;
        }
    }

public:
    // 构造函数，指定槽数和时间间隔
    TimeWheel(size_t slots, size_t interval_ms) 
        : slots_(slots), current_slot_(0), interval_(interval_ms), wheel_(slots), stop_(false), worker_(&TimeWheel::run, this) {}
    
    // 析构函数
    ~TimeWheel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }
    
    // 添加任务到时间轮
    void addTask(size_t delay_ms, Task task) {
        // 计算应该放入哪个槽
        size_t slot = (current_slot_ + (delay_ms + interval_ - 1) / interval_) % slots_;
        std::lock_guard<std::mutex> lock(mutex_);
        wheel_[slot].push_back(task);
    }
};

// 测试函数
void test_time_wheel() {
    // 创建一个具有10个槽、每个槽间隔100ms的时间轮
    TimeWheel tw(10, 100);
    
    // 添加一些测试任务
    tw.addTask(150, []() { std::cout << "Task 1 executed" << std::endl; });
    tw.addTask(250, []() { std::cout << "Task 2 executed" << std::endl; });
    tw.addTask(500, []() { std::cout << "Task 3 executed" << std::endl; });
    
    // 等待一段时间以观察任务执行
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

int main() {
    test_time_wheel();
    return 0;
}
