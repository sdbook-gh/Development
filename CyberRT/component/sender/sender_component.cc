#include "component/sender/sender_component.h"

namespace apollo {

bool Sender::Init() {
  ACHECK(ComponentBase::GetProtoConfig(&config_))
      << "failed to load sender config file "
      << ComponentBase::ConfigFilePath();
  AINFO << "Load config succedded.\n" << config_.DebugString();
  sender_writer_ = node_->CreateWriter<CarMsg>(config_.sender_topic().c_str());
  AINFO << "Init Sender succedded.";
  return true;
}

bool Sender::Proc() {
  AINFO << "Proc Sender triggered.";
  auto out_msg = std::make_shared<CarMsg>();
  static uint64_t speed = 1;
  out_msg->set_speed(speed++);
  out_msg->set_type("3");
  out_msg->set_plate("2");
  out_msg->set_owner("1");
  out_msg->set_kilometers(100);
  AINFO << "speed:" << speed;
  sender_writer_->Write(out_msg);
  return true;
}

} // namespace apollo