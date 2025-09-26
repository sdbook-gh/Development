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
  std::cout << diff.count() << '\n';
  auto diff_minutes = std::chrono::duration_cast<std::chrono::minutes>(diff);
  std::int64_t epoch_diff_minutes = diff_minutes.count();
  std::cout << epoch_diff_minutes << '\n';

  using namespace std::chrono_literals;

  auto y = 2025;
  auto m = 9u;
  auto d = 24u;
  int hh = 14, mm = 30, ss = 45;
  std::chrono::year_month_day ymd{std::chrono::year{y}, std::chrono::month{m}, std::chrono::day{d}};
  // 非法日期会在这里隐式转换成 ok()/sys_days 时暴露
  const std::chrono::sys_days dp = ymd; // days since 1970-01-01
  auto h = std::chrono::hours{hh};
  auto min = std::chrono::minutes{mm};
  auto s = std::chrono::seconds{ss};
  auto time_of_day = h + min + s; // duration 类型
  // auto tp = std::chrono::system_clock::time_point{dp} + time_of_day;
  auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>{dp} + time_of_day;
  std::cout << "timestamp (s): " << tp.time_since_epoch().count() << '\n';
  return 0;
}
