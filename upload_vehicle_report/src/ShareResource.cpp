#include "ShareResource.h"

using namespace v2x;

std::string ShareResource::configFilePath{"cfg.yaml"};
v2x::Config ShareResource::config;
std::mutex ShareResource::configMutex;

v2x::AtomicDequeue<ados_vehicle_msgs::msg::VehicleReport> ShareResource::vehicleReportQueue;
