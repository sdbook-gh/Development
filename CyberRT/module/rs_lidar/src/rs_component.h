#pragma once

#include <memory>
#include <atomic>
#include <string>
#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "rs_driver/api/lidar_driver.hpp"
#include "rs_driver/driver/driver_param.hpp"
#include "rs_driver/msg/packet.hpp"
#include "rs_driver/msg/point_cloud_msg.hpp"
#include "module/rs_lidar/proto/rs.pb.h"
#include "module/rs_lidar/proto/rs_config.pb.h"
#include "modules/common_msgs/sensor_msgs/pointcloud.pb.h"

namespace apollo {
namespace drivers {
namespace lidar {

using apollo::drivers::rs::Config;
using apollo::drivers::rs::LidarConfigBase;
using apollo::drivers::rs::LidarConfigBase_SourceType_RAW_PACKET;
using apollo::drivers::rs::LidarConfigBase_SourceType_ONLINE_LIDAR;
using PointT = PointXYZIRT ;
using PointCloudMsg = PointCloudT<PointT>;
using ::robosense::lidar::InputType;

class RsComponent : public cyber::Component<> {
 public:
  bool Init() override;
  void ReadScanCallback(const std::shared_ptr<robosense::RobosenseScanPacket>& scan_message);
  void RsPacketCallback(const ::robosense::lidar::Packet& lidar_packet);
  std::shared_ptr<PointCloudMsg> RsCloudAllocateCallback();
  void RsCloudCallback(std::shared_ptr<PointCloudMsg> rs_cloud);
  void PreparePointsMsg(PointCloud& msg);
  void ProcessCloud();

 private:
  std::shared_ptr<::robosense::lidar::LidarDriver<PointCloudMsg>> driver_ptr_;
  Config conf_;
  std::string frame_id_{""};
  std::atomic<int> pcd_sequence_num_{0};
  std::shared_ptr<cyber::Writer<robosense::RobosenseScanPacket>> scan_writer_{nullptr};
  std::shared_ptr<cyber::Reader<robosense::RobosenseScanPacket>> scan_reader_{nullptr};
  std::shared_ptr<cyber::Writer<PointCloud>> pcd_writer_{nullptr};

  bool InitConverter(const LidarConfigBase& lidar_config_base);
  bool InitPacket(const LidarConfigBase& lidar_config_base);
  bool InitBase(const LidarConfigBase& lidar_config_base);
  bool WriteScan(const std::shared_ptr<robosense::RobosenseScanPacket>& scan_message);
  bool WritePointCloud(const std::shared_ptr<PointCloud>& point_cloud);
};

CYBER_REGISTER_COMPONENT(RsComponent)

}  // namespace lidar
}  // namespace drivers
}  // namespace apollo
