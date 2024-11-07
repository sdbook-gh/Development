#include "component/receiver/receiver_component.h"

namespace apollo {

bool Receiver::Init() {

  ACHECK(ComponentBase::GetProtoConfig(&config_))
      << "failed to load receiver config file "
      << ComponentBase::ConfigFilePath();

  AINFO << "Load config succedded.\n" << config_.DebugString();
  control_writer_ =
      node_->CreateWriter<ControlMsg>(config_.control_topic().c_str());
  AINFO << "Init Receiver succedded.";
  return true;
}

bool Receiver::Proc(const std::shared_ptr<apollo::CarMsg> &msg0) {
  AINFO << "message recieved.\n" << msg0->DebugString();
  AINFO << "receive speed:" << msg0->speed();
  auto out_msg = std::make_shared<ControlMsg>();
  out_msg->set_speed(msg0->speed());
  AINFO << "receive speed:" << out_msg->speed();
  control_writer_->Write(out_msg);
  return true;
}

} // namespace apollo