#pragma once
#include <memory>

#include "component/sender/proto/sender.pb.h"
#include "cyber/common/macros.h"
#include "cyber/component/component.h"
#include "cyber/component/timer_component.h"
#include "cyber/cyber.h"

namespace apollo {
using apollo::cyber::Time;
using apollo::cyber::Writer;
class Sender final : public apollo::cyber::TimerComponent {
public:
  bool Init() override;
  bool Proc() override;

private:
  apollo::SenderConfig config_;
  std::shared_ptr<Writer<CarMsg>> sender_writer_ = nullptr;
};

CYBER_REGISTER_COMPONENT(Sender)

} // namespace apollo