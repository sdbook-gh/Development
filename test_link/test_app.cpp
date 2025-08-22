#include <cstdio>
#include <glog/logging.h>

#include "backward.hpp"

#if defined(TEST_LIB_STATIC)
int func_a();
#elif defined(TEST_LIB_DYNAMIC)
int func_so();
#endif
namespace backward {
backward::SignalHandling sh;
}

int main() {
#if defined(TEST_LIB_STATIC)
  func_a();
#elif defined(TEST_LIB_DYNAMIC)
  func_so();
#endif
  google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;
  google::SetStderrLogging(google::GLOG_INFO);
  LOG(ERROR) << "main";
  return 0;
}
