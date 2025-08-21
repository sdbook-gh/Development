#include "test_lib_so.h"
#include <glog/logging.h>
#include <exception>

int func_extra();
int func_so() {
  // google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;   
  google::SetStderrLogging(google::GLOG_INFO);  
  LOG(INFO) << "func_so";
  func_extra();
  // throw std::runtime_error{"test"};
  return 0;
}
