#include <cloud-msg/MessageDefinition.h>
#include "VehicleReportReceiver.h"
#include "Utils.h"
#include "ShareResource.h"

using namespace v2x;

extern Config config;

VehicleReportReceiver::VehicleReportReceiver(const std::string nodeName) : rclcpp::Node(nodeName)
{
	{
		std::unique_lock<std::mutex> lock(ShareResource::configMutex);
		vehiclereport_history_size = ShareResource::config.getYAMLConfigItem(VEHICLE)["vehiclereport_history_size"].as<int>();
	}
	PRINT_INFO("VehicleReportReceiver created");
	declare_parameter<int>("qos_queue_count", 10);
	const int qos_queue_count = get_parameter("qos_queue_count").as_int();
	subscription = create_subscription<ados_vehicle_msgs::msg::VehicleReport>("/ex/vehicle/report", qos_queue_count, std::bind(&VehicleReportReceiver::vehicleReport_callback, this, std::placeholders::_1));
}

void
VehicleReportReceiver::vehicleReport_callback(ados_vehicle_msgs::msg::VehicleReport::SharedPtr vehicleReport)
{
	if (ShareResource::vehicleReportQueue.size() >= vehiclereport_history_size)
	{
		ShareResource::vehicleReportQueue.popHead();
	}
	ShareResource::vehicleReportQueue.pushTail(*vehicleReport);
	PRINT_INFO("receive VehicleReport");
}
