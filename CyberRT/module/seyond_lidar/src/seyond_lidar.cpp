#include "module/seyond_lidar/src/seyond_lidar.h"

namespace apollo {
namespace drivers {
namespace lidar {

bool SeyondComponent::Init() {
  if (!GetProtoConfig(&conf_)) {
    AERROR << "load config error, file:" << config_file_path_;
    return false;
  }
  InitBase(conf_.config_base());
  driver_ptr_ = std::make_shared<SeyondDriver>();
  SeyondParam driver_param;
  driver_param.device_ip = conf_.device_ip();
  driver_param.port = conf_.port();
  driver_param.udp_port = conf_.udp_port();
  driver_param.reflectance_mode = conf_.reflectance_mode();
  driver_param.multiple_return = conf_.multiple_return();
  driver_param.coordinate_mode = conf_.coordinate_mode();
  driver_param.max_range = conf_.max_range();
  driver_param.min_range = conf_.min_range();
  driver_param.log_level = conf_.log_level();
  if (conf_.config_base().source_type() == LidarConfigBase_SourceType_RAW_PACKET) {
    driver_param.raw_packets_mode = true;
  }
  driver_ptr_->register_log_callback(&SeyondComponent::SeyondLogCallback);
  driver_ptr_->register_publish_packet_callback(std::bind(&SeyondComponent::SeyondPacketCallback, this, std::placeholders::_1, std::placeholders::_2));
  driver_ptr_->register_publish_point_callback(std::bind(&SeyondComponent::SeyondCloudCallback, this, std::placeholders::_1), std::bind(&SeyondComponent::SeyondCloudAllocateCallback, this));
  driver_param.print();
  scan_packets_ptr_ = std::make_shared<seyond::SeyondScan>();
  if (!driver_ptr_->init(driver_param)) {
    AERROR << "seyond Driver init failed";
    return false;
  }
  if (!driver_ptr_->start()) {
    AERROR << "seyond Driver start failed";
    return false;
  }
  AINFO << "seyond lidar init finished";
  return true;
}

void SeyondComponent::ReadScanCallback(const std::shared_ptr<seyond::SeyondScan>& scan_message) {
  static bool result = []{AINFO << "ReadScanCallback";return true;}();
  driver_ptr_->process_scan_packet_(scan_message);
}

void SeyondComponent::SeyondCloudCallback(std::shared_ptr<PointCloud> cloud) {
  static bool result = []{AINFO << "SeyondCloudCallback";return true;}();
  WritePointCloud(cloud);
}

std::shared_ptr<PointCloud> SeyondComponent::SeyondCloudAllocateCallback() {
  static bool result = []{AINFO << "SeyondCloudAllocateCallback";return true;}();
  return std::make_shared<PointCloud>();
}

void SeyondComponent::SeyondPacketCallback(const InnoDataPacket *pkt, bool is_next_frame) {
  static bool result = []{AINFO << "SeyondPacketCallback";return true;}();
  if (is_next_frame) {
    frame_count_++;
    scan_packets_ptr_->set_timestamp((static_cast<uint64_t>(pkt->common.ts_start_us)) * 1000);
    scan_packets_ptr_->set_measurement_time(pkt->common.ts_start_us * 1e-6);
    WriteScan(scan_packets_ptr_);
    scan_packets_ptr_ = std::make_shared<seyond::SeyondScan>();
  }
  seyond::SeyondPacket *scan_packet = scan_packets_ptr_->add_packets();
  uint64_t pkt_len = sizeof(InnoDataPacket) + pkt->item_number * pkt->item_size;
  scan_packet->mutable_data()->assign(reinterpret_cast<const char *>(pkt), pkt_len);
  scan_packet->set_table_exist(false);
  if (driver_ptr_->anglehv_table_init_ && (frame_count_ == table_send_hz_)) {
    frame_count_ = 0;
    scan_packet->set_table_exist(true);
    scan_packet->mutable_table()->assign(driver_ptr_->anglehv_table_.data(), driver_ptr_->anglehv_table_.size());
  }
}

void SeyondComponent::SeyondLogCallback(int32_t level, const char *header, const char *msg) {
  static bool result = []{AINFO << "SeyondLogCallback";return true;}();
  switch (level) {
    case 0:  // INNO_LOG_LEVEL_FATAL
    case 1:  // INNO_LOG_LEVEL_CRITICAL
    case 2:  // INNO_LOG_LEVEL_ERROR
    case 3:  // INNO_LOG_LEVEL_TEMP
      AERROR << header << " " << msg;
      break;
    case 4:  // INNO_LOG_LEVEL_WARNING
    case 5:  // INNO_LOG_LEVEL_DEBUG
      AWARN << header << " " << msg;
      break;
    default:
      AINFO << header << " " << msg;
  }
}

bool SeyondComponent::InitConverter(const LidarConfigBase& lidar_config_base) {
  pcd_writer_ = this->node_->CreateWriter<PointCloud>(lidar_config_base.point_cloud_channel());
  if(pcd_writer_ == nullptr) {
    return false;
  }
  if (lidar_config_base.source_type() == LidarConfigBase_SourceType_RAW_PACKET) {
    scan_reader_ = this->node_->CreateReader<seyond::SeyondScan>(lidar_config_base.scan_channel(), std::bind(&SeyondComponent::ReadScanCallback, this, std::placeholders::_1));
    if(scan_reader_ == nullptr) {
      return false;
    }
  }
  frame_id_ = lidar_config_base.frame_id();
  return true;
}

bool SeyondComponent::InitPacket(const LidarConfigBase& lidar_config_base) {
  if (lidar_config_base.source_type() == LidarConfigBase_SourceType_ONLINE_LIDAR) {
    scan_writer_ = this->node_->template CreateWriter<seyond::SeyondScan>(lidar_config_base.scan_channel());
    if(scan_writer_ == nullptr) {
      return false;
    }
  }
  return true;
}

bool SeyondComponent::InitBase(const LidarConfigBase& lidar_config_base) {
    if(!InitPacket(lidar_config_base)) {
      return false;
    }
    if(!InitConverter(lidar_config_base)) {
      return false;
    }
    return true;
}

bool SeyondComponent::WriteScan(const std::shared_ptr<seyond::SeyondScan>& scan_message) {
  return scan_writer_->Write(scan_message);
}

bool SeyondComponent::WritePointCloud(const std::shared_ptr<PointCloud>& point_cloud) {
  point_cloud->mutable_header()->set_frame_id(frame_id_);
  point_cloud->mutable_header()->set_sequence_num(pcd_sequence_num_.fetch_add(1));
  point_cloud->mutable_header()->set_timestamp_sec(cyber::Time().Now().ToSecond());
  if(!pcd_writer_->Write(point_cloud)) {
    return false;
  }
  return true;
}

}  // namespace lidar
}  // namespace drivers
}  // namespace apollo
