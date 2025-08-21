#include <cstdio>
#include <glog/logging.h>

#include "backward.hpp"

// int func_a();
int func_so();
namespace backward {
  backward::SignalHandling sh;
}

int main() {
  // func_a();
  func_so();
  google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;   
  google::SetStderrLogging(google::GLOG_INFO);  
  LOG(ERROR) << "main";
  return 0;
}
