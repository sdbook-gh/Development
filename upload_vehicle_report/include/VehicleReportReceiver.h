#ifndef UPLOAD_VEHICLE_REPORT_SRC_VEHICLEREPORTUPLOADER_H
#define UPLOAD_VEHICLE_REPORT_SRC_VEHICLEREPORTUPLOADER_H

#include "rclcpp/rclcpp.hpp"
#include "ados_vehicle_msgs/msg/vehicle_report.hpp"
#include <memory>

namespace v2x
{
class VehicleReportReceiver : public rclcpp::Node
{
public:
	explicit VehicleReportReceiver(const std::string nodeName);
private:
	std::shared_ptr<rclcpp::Subscription<ados_vehicle_msgs::msg::VehicleReport>> subscription;
	int vehiclereport_history_size;
	void
	vehicleReport_callback(ados_vehicle_msgs::msg::VehicleReport::SharedPtr vehicleReport);
};
}
#endif //UPLOAD_VEHICLE_REPORT_SRC_VEHICLEREPORTUPLOADER_H
