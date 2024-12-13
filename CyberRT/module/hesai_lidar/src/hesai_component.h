#pragma once

#include <memory>
#include <atomic>
#include <string>
#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "hesai2/hesai_lidar_sdk.hpp"
#include "hesai2/Lidar/lidar_types.h"
#include "module/hesai_lidar/proto/hesai.pb.h"
#include "module/hesai_lidar/proto/hesai_config.pb.h"
#include "modules/common_msgs/sensor_msgs/pointcloud.pb.h"

namespace apollo {
namespace drivers {
namespace lidar {

using apollo::drivers::hesai::Config;
using apollo::drivers::hesai::LidarConfigBase;
using apollo::drivers::hesai::LidarConfigBase_SourceType_RAW_PACKET;
using apollo::drivers::hesai::LidarConfigBase_SourceType_ONLINE_LIDAR;

class HesaiComponent : public cyber::Component<> {
 public:
  bool Init() override;
  void ReadScanCallback(const std::shared_ptr<HesaiUdpFrame>& scan_message);
  // Used to publish point clouds through 'ros_send_point_cloud_topic'
  void SendPointCloud(const LidarDecodedFrame<LidarPointXYZIRT>& msg);
  // Used to publish the original pcake through 'ros_send_packet_topic'
  void SendPacket(const UdpFrame_t& hesai_raw_msg, double);

 private:
  std::shared_ptr<HesaiLidarSdk<LidarPointXYZIRT>> driver_ptr_;
  Config conf_;
  int convert_threads_num_ = 1;
  std::shared_ptr<HesaiUdpFrame> scan_packets_ptr_{nullptr};
  std::string frame_id_{""};
  std::atomic<int> pcd_sequence_num_{0};
  std::shared_ptr<cyber::Writer<HesaiUdpFrame>> scan_writer_{nullptr};
  std::shared_ptr<cyber::Reader<HesaiUdpFrame>> scan_reader_{nullptr};
  std::shared_ptr<cyber::Writer<PointCloud>> pcd_writer_{nullptr};

  bool InitConverter(const LidarConfigBase& lidar_config_base);
  bool InitPacket(const LidarConfigBase& lidar_config_base);
  bool InitBase(const LidarConfigBase& lidar_config_base);
  bool WriteScan(const std::shared_ptr<HesaiUdpFrame>& scan_message);
  bool WritePointCloud(const std::shared_ptr<PointCloud>& point_cloud);
};

CYBER_REGISTER_COMPONENT(HesaiComponent)

}  // namespace lidar
}  // namespace drivers
}  // namespace apollo
