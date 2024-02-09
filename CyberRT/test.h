#include "proto/examples.pb.h"

#include <memory>
#include "cyber/component/component.h"

using apollo::cyber::Component;
using apollo::cyber::ComponentBase;
using apollo::cyber::examples::proto::Chatter;

class Test : public Component<Chatter> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Chatter>& msg) override;
};
CYBER_REGISTER_COMPONENT(Test)
