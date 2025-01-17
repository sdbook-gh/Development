#include "test_lib.h"
#include <glog/logging.h>
#include <exception>

int func1() {
  // google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;   
  google::SetStderrLogging(google::GLOG_INFO);  
  LOG(INFO) << "func1";
  throw std::runtime_error{"test"};
  return 0;
}
