/******************************************************************************
 * @file rtsp_camera_trigger_component.cc
 *****************************************************************************/

#include <cyber/cyber.h>
#include <gflags/gflags.h>
#include <chrono>
#include "module/rtsp_camera_trigger/proto/rtsp_camera_trigger.pb.h"

DEFINE_string(trigger_topic, "/camera/rtsp_camera_trigger", "trigger topic");

int main(int argc, const char* argv[]) {
  apollo::cyber::Init(argv[0]);
  auto node = apollo::cyber::CreateNode("rtsp_camera_trigger");
  auto client = node->CreateClient<apollo::RtspCameraTriggerRequest, apollo::RtspCameraTriggerResponse>(FLAGS_trigger_topic);
  auto request{std::make_shared<apollo::RtspCameraTriggerRequest>()};
  request->set_action("capture");
  auto response = client->SendRequest(request, std::chrono::seconds{5});
  if (response != nullptr) {
    AINFO << response->result();
  } else {
    AINFO << "service may not ready.";
  }
  return 0;
}
