#include "module/hesai_lidar/src/hesai_component.h"
#include "boost/format.hpp"

namespace apollo {
namespace drivers {
namespace lidar {

static uint64_t GetNanosecondTimestampFromSecondTimestamp(double second_timestamp) {
  auto ll_i = static_cast<uint64_t>(second_timestamp);
  uint64_t ll_f = (second_timestamp - ll_i) * 1e9;
  return ll_i * 1000000000LL + ll_f;
}

static double GetSecondTimestampFromNanosecondTimestamp(uint64_t nanosecond_timestamp) {
  uint64_t ll_i = nanosecond_timestamp / 1000000000ULL;
  uint64_t ll_f = nanosecond_timestamp - ll_i * 1000000000ULL;
  double d_f = ll_f * 1e-9;
  return static_cast<double>(ll_i) + d_f;
}

bool HesaiComponent::Init() {
  if (!GetProtoConfig(&conf_)) {
      AERROR << "load config error, file:" << config_file_path_;
      return false;
  }
  this->InitBase(conf_.config_base());

  if (conf_.has_convert_thread_nums()) {
      convert_threads_num_ = conf_.convert_thread_nums();
  }

  driver_ptr_ = std::make_shared<HesaiLidarSdk<LidarPointXYZIRT>>();

  DriverParam param;
  param.input_param.source_type = static_cast<SourceType>(conf_.source_type());
  param.input_param.pcap_path = conf_.pcap_path();
  param.input_param.correction_file_path = conf_.correction_file_path();
  param.input_param.firetimes_path = conf_.firetimes_path();
  param.input_param.device_ip_address = conf_.device_ip();
  param.input_param.host_ip_address = conf_.host_ip();
  param.input_param.ptc_port = conf_.ptc_port();
  param.input_param.udp_port = conf_.udp_port();
  param.input_param.multicast_ip_address = "";
  param.input_param.read_pcap = true;

  param.decoder_param.pcap_play_synchronization = false;
  param.decoder_param.frame_start_azimuth = conf_.frame_start_azimuth();
  param.decoder_param.enable_parser_thread = true;

  param.decoder_param.transform_param.x = 0;
  param.decoder_param.transform_param.y = 0;
  param.decoder_param.transform_param.z = 0;
  param.decoder_param.transform_param.pitch = 0;
  param.decoder_param.transform_param.yaw = 0;
  param.decoder_param.transform_param.roll = 0;

  param.lidar_type = conf_.lidar_type();

  driver_ptr_->RegRecvCallback(std::bind(&HesaiComponent::SendPointCloud, this, std::placeholders::_1));

  if (conf_.config_base().source_type() == LidarConfigBase_SourceType_RAW_PACKET) {
    param.decoder_param.enable_udp_thread = false;
  }

  if (conf_.config_base().source_type() == LidarConfigBase_SourceType_ONLINE_LIDAR) {
    driver_ptr_->RegRecvCallback(std::bind(&HesaiComponent::SendPacket,this,std::placeholders::_1,std::placeholders::_2));
  }

  if (!driver_ptr_->Init(param)) {
    AERROR << "init fail";
    return false;
  }
  driver_ptr_->Start();
  AINFO << "hesai lidar init finished";
  return true;
}

void HesaiComponent::ReadScanCallback(const std::shared_ptr<HesaiUdpFrame>& scan_message) {
  // UdpPacket packet;
  // UdpFrame_t frame_t;
  for (int i = 0, packet_size = scan_message->packets_size(); i < packet_size; ++i) {
    driver_ptr_->lidar_ptr_->origin_packets_buffer_.emplace_back(reinterpret_cast<const uint8_t*>(scan_message->packets(i).data().c_str()), scan_message->packets(i).data().length());
  }
}

void HesaiComponent::SendPointCloud(const LidarDecodedFrame<LidarPointXYZIRT>& msg) {
  std::shared_ptr<PointCloud> cloud_message = std::make_shared<PointCloud>();
  cloud_message->set_is_dense(false);
  cloud_message->mutable_header()->set_timestamp_sec(cyber::Time().Now().ToSecond());
  double timestamp_diff = 0.0;
  int point_size = msg.points_num;
  if (point_size > 0) {
    const double pcl_timestamp = msg.points[point_size - 1].timestamp;
    cloud_message->set_measurement_time(pcl_timestamp);
    timestamp_diff = pcl_timestamp - msg.points[0].timestamp;
  } else {
    cloud_message->set_measurement_time(0.0);
  }
  for (int i = 0; i < point_size; ++i) {
    cloud_message->add_point();
  }
#pragma omp parallel for schedule(static) num_threads(convert_threads_num_)
  for (int i = 0; i < point_size; ++i) {
    cloud_message->mutable_point(i)->set_x(msg.points[i].x);
    cloud_message->mutable_point(i)->set_y(msg.points[i].y);
    cloud_message->mutable_point(i)->set_z(msg.points[i].z);
    cloud_message->mutable_point(i)->set_timestamp(GetNanosecondTimestampFromSecondTimestamp(msg.points[i].timestamp));
    cloud_message->mutable_point(i)->set_intensity(msg.points[i].intensity);
  }
  AINFO << boost::format("point cnt = %d; timestamp_diff = %.9f s") % point_size % timestamp_diff;
  this->WritePointCloud(cloud_message);
}

void HesaiComponent::SendPacket(const UdpFrame_t& hesai_raw_msg, double timestamp) {
  std::shared_ptr<HesaiUdpFrame> udp_frame = std::make_shared<HesaiUdpFrame>();
  for (auto i = 0u; i < hesai_raw_msg.size(); ++i) {
    auto packet = udp_frame->add_packets();
    packet->set_timestamp_sec(timestamp);
    packet->set_size(hesai_raw_msg[i].packet_len);
    packet->mutable_data()->assign(reinterpret_cast<const char*>(hesai_raw_msg[i].buffer), reinterpret_cast<const char*>(hesai_raw_msg[i].buffer) + hesai_raw_msg[i].packet_len);
  }
  this->WriteScan(udp_frame);
}

bool HesaiComponent::InitConverter(const LidarConfigBase& lidar_config_base) {
  pcd_writer_ = this->node_->CreateWriter<PointCloud>(lidar_config_base.point_cloud_channel());
  if(pcd_writer_ == nullptr) {
    return false;
  }
  if (lidar_config_base.source_type() == 1) {
    scan_reader_ = this->node_->CreateReader<HesaiUdpFrame>(lidar_config_base.scan_channel(), std::bind(&HesaiComponent::ReadScanCallback, this, std::placeholders::_1));
    if(scan_reader_ == nullptr) {
      return false;
    }
  }
  frame_id_ = lidar_config_base.frame_id();
  return true;
}

bool HesaiComponent::InitPacket(const LidarConfigBase& lidar_config_base) {
  if (lidar_config_base.source_type() == 0) {
    scan_writer_ = this->node_->template CreateWriter<HesaiUdpFrame>(lidar_config_base.scan_channel());
    if(scan_writer_ == nullptr) {
      return false;
    }
  }
  return true;
}

bool HesaiComponent::InitBase(const LidarConfigBase& lidar_config_base) {
    if(!InitPacket(lidar_config_base)) {
      return false;
    }
    if(!InitConverter(lidar_config_base)) {
      return false;
    }
    return true;
}

bool HesaiComponent::WriteScan(const std::shared_ptr<HesaiUdpFrame>& scan_message) {
  return scan_writer_->Write(scan_message);
}

bool HesaiComponent::WritePointCloud(const std::shared_ptr<PointCloud>& point_cloud) {
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
