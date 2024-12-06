/******************************************************************************
 * @file rtsp_camera_component.h
 *****************************************************************************/

#pragma once
#include <memory>

#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "module/rtsp_camera_trigger/proto/rtsp_camera_trigger.pb.h"

namespace apollo {

class RtspCamera final : public cyber::Component<> {
 public:
  bool Init() override;

 private:
  int run();
  std::shared_ptr<apollo::cyber::Service<apollo::RtspCameraTriggerRequest, apollo::RtspCameraTriggerResponse>> service;
};

CYBER_REGISTER_COMPONENT(RtspCamera)

} // namespace apollo
