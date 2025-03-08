// open.cpp

#include "log.h"
#include <cstring>

open::nmutils::nmlog::NMAsyncLog::NMAsyncLog()
    : NMAsyncLog(1) {} // Default constructor with 1 worker thread

open::nmutils::nmlog::NMAsyncLog::NMAsyncLog(int numThreads) : m_stopFlag(false) {
  for (int i = 0; i < numThreads; i++) {
    m_logThreads.emplace_back(&NMAsyncLog::logThread, this);
  }
}

open::nmutils::nmlog::NMAsyncLog::~NMAsyncLog() {
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_stopFlag = true;
    m_cv.notify_all();
  }
  for (auto &thread : m_logThreads) {
    thread.join();
  }
}

void open::nmutils::nmlog::NMAsyncLog::log(const char *format, ...) {
  va_list args;
  va_start(args, format);
  char buffer[256];
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);

  if (std::strncmp(buffer, "error", 5) == 0) {
    for (const auto &handler : m_errorHandlers) {
      handler(buffer);
    }
  }

  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_logQueue.push(buffer);
    m_cv.notify_all();
  }
}

void open::nmutils::nmlog::NMAsyncLog::logThread() {
  while (true) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [this]() { return !m_logQueue.empty() || m_stopFlag; });

    if (m_stopFlag) {
      break;
    }

    std::string message = m_logQueue.front();
    m_logQueue.pop();

    lock.unlock();

    // Write the log message to the file
    m_logFile << message << std::endl;
  }
}

void open::nmutils::nmlog::NMAsyncLog::addErrorHandler(
    const ErrorHandler &handler) {
  m_errorHandlers.push_back(handler);
}
