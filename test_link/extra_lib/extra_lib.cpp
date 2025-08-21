#include "extra_lib.h"
#include <glog/logging.h>
#include <exception>

int func_extra() {
  // google::InitGoogleLogging("");
  FLAGS_logtostderr = 1;   
  google::SetStderrLogging(google::GLOG_INFO);  
  LOG(INFO) << "func_extra";
  return 0;
}
