#include "examples.pb.h"

#include <memory>
#include "cyber/component/component.h"

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using testproto::Driver;

class CommonComponentSample : public Component<Driver, Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0,
            const std::shared_ptr<Driver>& msg1) override;
};
CYBER_REGISTER_COMPONENT(CommonComponentSample)
