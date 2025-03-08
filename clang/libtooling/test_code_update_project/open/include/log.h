#pragma once

#include <condition_variable>
#include <cstdarg>
#include <fstream>
#include <functional>
#include <mutex>
#include <vector>
#include <queue>
#include <sstream>
#include <string>
#include <thread>

namespace open {
namespace nmutils {
namespace nmlog {

class NMAsyncLog {
public:
  using ErrorHandler = std::function<void(const std::string &)>;
  NMAsyncLog();
  NMAsyncLog(int numThreads); // Constructor with the number of worker threads
  ~NMAsyncLog();

  void log(const char *format, ...);

  template <typename T>
  friend NMAsyncLog &operator<<(NMAsyncLog &log, const T &value);

  void addErrorHandler(const ErrorHandler &handler);

private:
  void logThread();

  std::ofstream m_logFile;
  std::queue<std::string> m_logQueue;
  std::mutex m_mutex;
  std::condition_variable m_cv;
  bool m_stopFlag;
  std::vector<std::thread> m_logThreads;
  std::vector<ErrorHandler> m_errorHandlers;
};

template <typename T> NMAsyncLog &operator<<(NMAsyncLog &log, const T &value) {
  std::ostringstream oss;
  oss << value;
  log.log(oss.str().c_str());
  return log;
}

} // namespace log
} // namespace utils
} // namespace open
