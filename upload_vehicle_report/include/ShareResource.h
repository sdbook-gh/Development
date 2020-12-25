#ifndef UPLOAD_VEHICLE_REPORT_SRC_SHARERESOURCE_H
#define UPLOAD_VEHICLE_REPORT_SRC_SHARERESOURCE_H

#include "ados_vehicle_msgs/msg/vehicle_report.hpp"
#include "Utils.h"
#include <deque>
#include <mutex>
#include "AtomicDequeue.h"

namespace v2x
{
class ShareResource
{
public:
	static std::string configFilePath;
	static v2x::Config config;
	static std::mutex configMutex;
	static v2x::AtomicDequeue<ados_vehicle_msgs::msg::VehicleReport> vehicleReportQueue;
};
}

#endif //UPLOAD_VEHICLE_REPORT_SRC_SHARERESOURCE_H
