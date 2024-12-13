#pragma once

#include <memory>
#include <atomic>
#include <string>
#include "cyber/cyber.h"
#include "cyber/component/component.h"
#include "module/seyond_lidar/src/seyond_driver.h"
#include "module/seyond_lidar/proto/seyond.pb.h"
#include "module/seyond_lidar/proto/seyond_config.pb.h"
#include "modules/common_msgs/sensor_msgs/pointcloud.pb.h"

namespace apollo {
namespace drivers {
namespace lidar {

using apollo::drivers::seyond::Config;
using apollo::drivers::seyond::LidarConfigBase;
using apollo::drivers::seyond::LidarConfigBase_SourceType_RAW_PACKET;
using apollo::drivers::seyond::LidarConfigBase_SourceType_ONLINE_LIDAR;

class SeyondComponent : public cyber::Component<> {
 public:
  bool Init() override;
  void ReadScanCallback(const std::shared_ptr<seyond::SeyondScan>& scan_message);
  void SeyondCloudCallback(std::shared_ptr<PointCloud> cloud);
  void SeyondPacketCallback(const InnoDataPacket *pkt, bool is_next_frame);
  std::shared_ptr<PointCloud> SeyondCloudAllocateCallback();
  static void SeyondLogCallback(int32_t level, const char *header, const char *msg);

 private:
  std::shared_ptr<SeyondDriver> driver_ptr_;
  Config conf_;
  std::shared_ptr<seyond::SeyondScan> scan_packets_ptr_{nullptr};
  uint32_t table_send_hz_{10};
  uint32_t frame_count_{0};
  std::string frame_id_{""};
  std::atomic<int> pcd_sequence_num_{0};
  std::shared_ptr<cyber::Writer<seyond::SeyondScan>> scan_writer_{nullptr};
  std::shared_ptr<cyber::Reader<seyond::SeyondScan>> scan_reader_{nullptr};
  std::shared_ptr<cyber::Writer<PointCloud>> pcd_writer_{nullptr};

  bool InitConverter(const LidarConfigBase& lidar_config_base);
  bool InitPacket(const LidarConfigBase& lidar_config_base);
  bool InitBase(const LidarConfigBase& lidar_config_base);
  bool WriteScan(const std::shared_ptr<seyond::SeyondScan>& scan_message);
  bool WritePointCloud(const std::shared_ptr<PointCloud>& point_cloud);
};

CYBER_REGISTER_COMPONENT(SeyondComponent)

}  // namespace lidar
}  // namespace drivers
}  // namespace apollo
