#include "test.h"

bool Test::Init() {
  AINFO << "Test init";
  return true;
}

bool Test::Proc(const std::shared_ptr<Chatter>& msg) {
  AINFO << "Test receive [" << msg->seq() << "]";
  return true;
}
