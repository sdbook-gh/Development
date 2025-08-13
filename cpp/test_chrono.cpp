#include <chrono>
#include <ctime>
#include <iostream>

int main() {
  std::tm tmw{0};
  tmw.tm_year = 1601 - 1900; // start from 1900
  tmw.tm_mon = 0; // start from 0
  tmw.tm_mday = 1;
  std::chrono::system_clock::time_point tpw = std::chrono::system_clock::from_time_t(std::mktime(&tmw));
  std::time_t time_tpw = std::chrono::system_clock::to_time_t(tpw);
  {
    std::tm local_time;
    localtime_r(&time_tpw, &local_time);
    std::cout << "年: " << local_time.tm_year + 1900 << std::endl; // tm_year是从1900年开始计数
    std::cout << "月: " << local_time.tm_mon + 1 << std::endl; // tm_mon的范围是0-11
    std::cout << "日: " << local_time.tm_mday << std::endl;
    std::cout << "时: " << local_time.tm_hour << std::endl;
    std::cout << "分: " << local_time.tm_min << std::endl;
    std::cout << "秒: " << local_time.tm_sec << std::endl;
  }
  std::tm tmu{0};
  tmu.tm_year = 1970 - 1900;
  tmu.tm_mon = 0;
  tmu.tm_mday = 1;
  std::chrono::system_clock::time_point tpu = std::chrono::system_clock::from_time_t(std::mktime(&tmu));
  std::time_t time_tpu = std::chrono::system_clock::to_time_t(tpu);
  {
    std::tm local_time;
    localtime_r(&time_tpu, &local_time);
    std::cout << "年: " << local_time.tm_year + 1900 << std::endl; // tm_year是从1900年开始计数
    std::cout << "月: " << local_time.tm_mon + 1 << std::endl; // tm_mon的范围是0-11
    std::cout << "日: " << local_time.tm_mday << std::endl;
    std::cout << "时: " << local_time.tm_hour << std::endl;
    std::cout << "分: " << local_time.tm_min << std::endl;
    std::cout << "秒: " << local_time.tm_sec << std::endl;
  }
  auto diff = tpu - tpw;
  auto diff_100ns = std::chrono::duration_cast<std::chrono::seconds>(diff);
  std::int64_t epoch_diff_100ns = diff_100ns.count();
  std::cout << epoch_diff_100ns << '\n';
  return 0;
}
