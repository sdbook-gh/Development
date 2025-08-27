#include <chrono>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
  // const auto start_time = std::chrono::steady_clock::now().time_since_epoch();
  // std::cout << start_time.count() << "\n";

  // if (argc < 2) { return 1; }
  // const auto start_time = std::chrono::steady_clock::now().time_since_epoch();
  // const auto real_time_stamp = std::chrono::system_clock::now();
  // const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(start_time);
  // long input_ms = std::stod(argv[1]) * 1000.0;
  // const auto diff_ms = input_ms - ms.count();
  // const auto result_time_stamp = real_time_stamp + std::chrono::milliseconds(diff_ms);
  // std::cout << "ms: " << ms.count() << '\n';
  // auto time_t_result = std::chrono::system_clock::to_time_t(result_time_stamp);
  // std::tm* tm_result = std::localtime(&time_t_result);
  // std::cout << "Date: " << std::put_time(tm_result, "%Y-%m-%d") << '\n';
  // std::cout << "Time: " << std::put_time(tm_result, "%H:%M:%S") << '\n';

  if (argc < 4) { return 1; }
  long input_real_time_ns = std::stoul(argv[1]);
  const auto real_time_stamp = std::chrono::system_clock::time_point(std::chrono::nanoseconds(input_real_time_ns));
  {
    auto time_t_result = std::chrono::system_clock::to_time_t(real_time_stamp);
    std::tm* tm_result = std::localtime(&time_t_result);
    std::cout << "Date: " << std::put_time(tm_result, "%Y-%m-%d") << '\n';
    std::cout << "Time: " << std::put_time(tm_result, "%H:%M:%S") << '\n';
  }
  long ms = std::stoul(argv[2]) / 1e6;
  long input_ms = std::stod(argv[3]) * 1e3;
  const auto diff_ms = input_ms - ms;
  std::cout << "diff_ms: " << diff_ms << '\n';
  const auto result_time_stamp = real_time_stamp + std::chrono::milliseconds(diff_ms);
  std::cout << "ms: " << ms << '\n';
  {
    auto time_t_result = std::chrono::system_clock::to_time_t(result_time_stamp);
    std::tm* tm_result = std::localtime(&time_t_result);
    std::cout << "Date: " << std::put_time(tm_result, "%Y-%m-%d") << '\n';
    std::cout << "Time: " << std::put_time(tm_result, "%H:%M:%S") << '\n';
  }
  return 0;
}
