#pragma once
#include <memory>

#include "component/receiver/proto/receiver.pb.h"
#include "component/sender/proto/sender.pb.h"
#include "cyber/component/component.h"
#include "cyber/cyber.h"
using apollo::cyber::Time;
using apollo::cyber::Writer;
namespace apollo {

class Receiver final : public cyber::Component<apollo::CarMsg> {
public:
  bool Init() override;
  bool Proc(const std::shared_ptr<apollo::CarMsg> &msg0) override;

private:
  apollo::ReceiverConfig config_;
  std::shared_ptr<Writer<ControlMsg>> control_writer_ = nullptr;
};

CYBER_REGISTER_COMPONENT(Receiver)

} // namespace apollo