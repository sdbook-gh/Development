#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

class Timer {
public:
  Timer() : start(std::chrono::high_resolution_clock::now()) {}

  double elapsed() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};

extern "C" void example_function(int iteration) {
  volatile double result = 0.0;
  long iterations = (iteration % 100 == 99) ? 10000000 : 100000;

  for (long i = 0; i < iterations; ++i) {
    result += std::sqrt(std::atan(i));
  }
}

int main() {
  std::vector<double> timings;
  int iteration = 0;

  Timer overall_timer;

  // 无限循环，记录函数执行时间
  while (true) {
    Timer timer;
    example_function(iteration);
    timings.push_back(timer.elapsed());
    iteration++;

    if (overall_timer.elapsed() >= 1000.0) {
      // 计算平均耗时
      double average =
          std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();

      // 计算 P99 耗时
      std::sort(timings.begin(), timings.end());
      double p99 = timings[static_cast<int>(timings.size() * 0.99)];

      std::cout << "Average execution time: " << average << " ms" << std::endl;
      std::cout << "P99 execution time: " << p99 << " ms" << std::endl;

      // 清空 timings 以便于下一次计算
      timings.clear();

      // 重置计时器
      overall_timer = Timer();
    }
  }

  return 0;
}