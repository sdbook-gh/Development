#include "module/rs_lidar/src/rs_component.h"

namespace apollo {
namespace drivers {
namespace lidar {

static uint64_t GetNanosecondTimestampFromSecondTimestamp(double second_timestamp) {
  auto ll_i = static_cast<uint64_t>(second_timestamp);
  uint64_t ll_f = (second_timestamp - ll_i) * 1e9;
  return ll_i * 1000000000LL + ll_f;
}

static double GetSecondTimestampFromNanosecondTimestamp(
  uint64_t nanosecond_timestamp) {
  uint64_t ll_i = nanosecond_timestamp / 1000000000ULL;
  uint64_t ll_f = nanosecond_timestamp - ll_i * 1000000000ULL;
  double d_f = ll_f * 1e-9;
  return static_cast<double>(ll_i) + d_f;
}

bool RsComponent::Init() {
    if (!GetProtoConfig(&conf_)) {
        AERROR << "load config error, file:" << config_file_path_;
        return false;
    }

    this->InitBase(conf_.config_base());

    driver_ptr_ = std::make_shared<::robosense::lidar::LidarDriver<PointCloudMsg>>();

    ::robosense::lidar::RSDecoderParam decoder_param;
    decoder_param.min_distance = conf_.min_distance();
    decoder_param.max_distance = conf_.max_distance();
    decoder_param.start_angle = conf_.start_angle();
    decoder_param.end_angle = conf_.end_angle();
    decoder_param.use_lidar_clock = conf_.use_lidar_clock();
    decoder_param.num_blks_split = conf_.num_pkts_split();
    decoder_param.dense_points = conf_.dense_points();
    decoder_param.ts_first_point = conf_.ts_first_point();
    decoder_param.wait_for_difop = conf_.wait_for_difop();
    decoder_param.config_from_file = conf_.config_from_file();
    decoder_param.angle_path = conf_.angle_path();
    decoder_param.split_angle = conf_.split_angle();

    ::robosense::lidar::RSInputParam input_param;
    input_param.msop_port = conf_.msop_port();
    input_param.difop_port = conf_.difop_port();
    input_param.host_address = conf_.host_address();
    input_param.group_address = conf_.group_address();
    input_param.use_vlan = conf_.use_vlan();
    input_param.user_layer_bytes = conf_.user_layer_bytes();
    input_param.tail_layer_bytes = conf_.tail_layer_bytes();

    ::robosense::lidar::RSDriverParam driver_param;
    driver_param.input_param = input_param;
    driver_param.decoder_param = decoder_param;
    driver_param.lidar_type = ::robosense::lidar::strToLidarType(conf_.model());

    if (conf_.config_base().source_type() == LidarConfigBase_SourceType_RAW_PACKET) {
        driver_param.input_type = InputType::RAW_PACKET;
    } else if (conf_.config_base().source_type() == LidarConfigBase_SourceType_ONLINE_LIDAR) {
        driver_param.input_type = InputType::ONLINE_LIDAR;
        driver_ptr_->regPacketCallback(std::bind(&RsComponent::RsPacketCallback, this, std::placeholders::_1));
    }

    driver_ptr_->regPointCloudCallback(std::bind(&RsComponent::RsCloudAllocateCallback, this), std::bind(&RsComponent::RsCloudCallback, this,  std::placeholders::_1));
    driver_ptr_->regExceptionCallback([](const ::robosense::lidar::Error& code) { RS_WARNING << code.toString() << RS_REND; });
    driver_param.print();
    if (!driver_ptr_->init(driver_param)) {
        AERROR << "Robosense Driver init failed";
        return false;
    }

    if (!driver_ptr_->start()) {
        AERROR << "Robosense Driver start failed";
        return false;
    }
    AINFO << "RSComponent init finished";
    return true;
}

void RsComponent::ReadScanCallback(const std::shared_ptr<robosense::RobosenseScanPacket>& scan_message) {
    ADEBUG << __FUNCTION__ << " start";
    std::shared_ptr<::robosense::lidar::Packet> scan_packet = std::make_shared<::robosense::lidar::Packet>();
    scan_packet->buf_.assign(scan_message->data().begin(), scan_message->data().end());
    driver_ptr_->decodePacket(*scan_packet);
}

void RsComponent::RsPacketCallback(const ::robosense::lidar::Packet& lidar_packet) {
    ADEBUG << __FUNCTION__ << " start";
    auto scan_packet = std::make_shared<robosense::RobosenseScanPacket>();
    scan_packet->set_stamp(cyber::Time::Now().ToNanosecond());
    scan_packet->mutable_data()->assign(lidar_packet.buf_.begin(), lidar_packet.buf_.end());
    WriteScan(scan_packet);
}

std::shared_ptr<PointCloudMsg> RsComponent::RsCloudAllocateCallback() {
    return std::make_shared<PointCloudMsg>();
}

void RsComponent::RsCloudCallback(std::shared_ptr<PointCloudMsg> rs_cloud) {
  auto apollo_pc = std::make_shared<PointCloud>();
  for (auto p : rs_cloud->points) {
      PointXYZIT* point = apollo_pc->add_point();
      point->set_x(p.x);
      point->set_y(p.y);
      point->set_z(p.z);
      point->set_intensity(uint32_t(p.intensity));
      point->set_timestamp(GetNanosecondTimestampFromSecondTimestamp(p.timestamp));
  }
  PreparePointsMsg(*apollo_pc);
  WritePointCloud(apollo_pc);
}

void RsComponent::PreparePointsMsg(PointCloud& msg) {
    msg.set_height(1);
    msg.set_width(msg.point_size() / msg.height());
    msg.set_is_dense(false);
    const auto timestamp = msg.point(static_cast<int>(msg.point_size()) - 1).timestamp();
    msg.set_measurement_time(GetSecondTimestampFromNanosecondTimestamp(timestamp));
    double lidar_time = GetSecondTimestampFromNanosecondTimestamp(timestamp);
    double diff_time = msg.header().timestamp_sec() - lidar_time;
    if (diff_time > 0.2) {
        AINFO << std::fixed << std::setprecision(16) << "system time: " << msg.header().timestamp_sec() << ", lidar time: " << lidar_time << ", diff is:" << diff_time;
    }
    if (conf_.use_lidar_clock()) {
        msg.mutable_header()->set_lidar_timestamp(timestamp);
    } else {
        msg.mutable_header()->set_lidar_timestamp(cyber::Time().Now().ToNanosecond());
    }
}

bool RsComponent::InitConverter(const LidarConfigBase& lidar_config_base) {
  pcd_writer_ = this->node_->CreateWriter<PointCloud>(lidar_config_base.point_cloud_channel());
  if(pcd_writer_ == nullptr) {
    return false;
  }
  if (lidar_config_base.source_type() == 1) {
    scan_reader_ = this->node_->CreateReader<robosense::RobosenseScanPacket>(lidar_config_base.scan_channel(), std::bind(&RsComponent::ReadScanCallback, this, std::placeholders::_1));
    if(scan_reader_ == nullptr) {
      return false;
    }
  }
  frame_id_ = lidar_config_base.frame_id();
  return true;
}

bool RsComponent::InitPacket(const LidarConfigBase& lidar_config_base) {
  if (lidar_config_base.source_type() == 0) {
    scan_writer_ = this->node_->template CreateWriter<robosense::RobosenseScanPacket>(lidar_config_base.scan_channel());
    if(scan_writer_ == nullptr) {
      return false;
    }
  }
  return true;
}

bool RsComponent::InitBase(const LidarConfigBase& lidar_config_base) {
    if(!InitPacket(lidar_config_base)) {
      return false;
    }
    if(!InitConverter(lidar_config_base)) {
      return false;
    }
    return true;
}

bool RsComponent::WriteScan(const std::shared_ptr<robosense::RobosenseScanPacket>& scan_message) {
  return scan_writer_->Write(scan_message);
}

bool RsComponent::WritePointCloud(const std::shared_ptr<PointCloud>& point_cloud) {
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
