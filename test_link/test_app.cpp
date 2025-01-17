#include <cstdio>
#include "test_lib.h"
#include <glog/logging.h>

#include "backward.hpp"

namespace backward {
  backward::SignalHandling sh;
}

int main() {
  func1();
  google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;   
  google::SetStderrLogging(google::GLOG_INFO);  
  LOG(ERROR) << "main";
  return 0;
}