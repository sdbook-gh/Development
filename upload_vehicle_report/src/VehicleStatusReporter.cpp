#include "VehicleStatusReporter.h"
#include "Utils.h"
#include "cloud-msg/MessageDefinition.h"
#include <vector>
#include <thread>
#include <chrono>
#include <sys/prctl.h>
#include "ShareResource.h"

using namespace v2x;

#define SLEEP_MS 1000

bool
VehicleStatusReporter::startReportVehicleStatus(SSLClient &sslClientUplink, SSLClient &sslClientDownlink)
{
	std::string vehicleId;
	int send_timeout_second;
	int recv_timeout_second;
	{
		std::unique_lock<std::mutex> lock(ShareResource::configMutex);
		vehicleId = ShareResource::config.getYAMLConfigItem(GLOBAL)["vehicle_id"].as<std::string>();
		send_timeout_second = ShareResource::config.getYAMLConfigItem(GLOBAL)["socket_send_timeout"].as<int>();
		recv_timeout_second = ShareResource::config.getYAMLConfigItem(GLOBAL)["socket_recv_timeout"].as<int>();
	}

	std::thread th_VEH2CLOUD_RUN_sender([&sslClientUplink, &sslClientDownlink, vehicleId, send_timeout_second]() -> void
										{
											set_thread_name("th_VEH2CLOUD_RUN_sender");
											PRINT_INFO("th_VEH2CLOUD_RUN_sender started");

											while (true)
											{
												if (!sslClientUplink.isWorkable())
												{
													PRINT_ERROR("client status wrong");
													return;
												}
												Header header;
												VEH2CLOUD_RUN run(vehicleId);

												setValue(header.dataType, DataType::VEH2CLOUD_RUN);
												setValue(header.version, V2XPROTOCOL_VERSION);
												setValue(header.ts_ms, 0);
												setValue(header.ts_min, 0);

												ados_vehicle_msgs::msg::VehicleReport vehicleReport;
												if (ShareResource::vehicleReportQueue.popTail(vehicleReport))
												{
													ShareResource::vehicleReportQueue.clear();
													PRINT_INFO("sending VehicleReport");
													// default
													setValue(run.tapPos, 255);
													if (vehicleReport.gear_report == 3)
													{
														// D
														setValue(run.tapPos, 21);
													}
													else if (vehicleReport.gear_report == 1)
													{
														// R
														setValue(run.tapPos, 22);
													}
													// default
													setValue(run.steeringAngle, 0xFFFFFFFF);
													setValue(run.steeringAngle, vehicleReport.steering_wheel_angle);
													// default
													setValue(run.lights, 0);
													if (vehicleReport.turn_signal_report.value == 1)
													{
														setValue(run.lights, 1 << 2);
													}
													else if (vehicleReport.turn_signal_report.value == 2)
													{
														setValue(run.lights, 1 << 3);
													}
													setValue(run.velocityCAN, vehicleReport.velocity);
													setValue(run.acceleration_V, vehicleReport.longitudinal_acceleration);
													setValue(run.acceleration_H, vehicleReport.lateral_acceleration);
													setValue(run.accelPos, vehicleReport.throttle_override);
													setValue(run.engineSpeed, vehicleReport.engine_rpm);
													// default
													setValue(run.engineTorque, 65535);
													setValue(run.brakeFlag, vehicleReport.brake_override);
													// default
													setValue(run.brakePos, 255);
													// default
													setValue(run.brakePressure, 65535);
													setValue(run.yawRate, vehicleReport.yaw_rate);
													// convert from km/h to m/s
													setValue(run.wheelVelocity_FL, vehicleReport.wheel_speed_front_left / 3.6);
													// convert from km/h to m/s
													setValue(run.wheelVelocity_FR, vehicleReport.wheel_speed_front_right / 3.6);
													// convert from km/h to m/s
													setValue(run.wheelVelocity_RL, vehicleReport.wheel_speed_rear_left / 3.6);
													// convert from km/h to m/s
													setValue(run.wheelVelocity_RR, vehicleReport.wheel_speed_rear_right / 3.6);
													// default
													setValue(run.absFlag, 255);
													// default
													setValue(run.tcsFlag, 255);
													// default
													setValue(run.espFlag, 255);
													// default
													setValue(run.lkaFlag, 255);
													// default
													setValue(run.accMode, 255);
												}

												int run_len = run.calcRealSize();
												setValue(header.remainLength, run_len);
												int header_len = header.calcRealSize();
												int total_len = header_len + run_len;
												std::vector<uint8_t> sendbuffer;
												sendbuffer.resize(total_len);
												header.fillBuffer(&sendbuffer[0], 0, header_len);
												run.fillBuffer(&sendbuffer[0] + header_len, 0, run_len);
												if (sslClientUplink.send(&sendbuffer[0], 0, total_len, total_len, send_timeout_second) != total_len)
												{
													sslClientUplink.stopWork();
													sslClientDownlink.stopWork();
													PRINT_ERROR("send head to server failed");
													return;
												}
												else
												{
													static bool alreadySend = false;
													if (!alreadySend)
													{
														PRINT_INFO("send head to server ok");
														alreadySend = true;
													}
												}
												std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS));
											}
										});

	std::thread th_response_receiver([&sslClientUplink, &sslClientDownlink, recv_timeout_second]() -> void
									 {
										 set_thread_name("th_response_receiver");
										 PRINT_INFO("th_response_receiver started");

										 while (true)
										 {
											 if (!sslClientUplink.isWorkable())
											 {
												 PRINT_ERROR("client status wrong");
												 return;
											 }
											 std::vector<uint8_t> recvbuffer;
											 Header rcvHeader;
											 int header_len = rcvHeader.calcRealSize();
											 recvbuffer.resize(header_len);
											 // endless wait for response
											 int recv_len = sslClientUplink.receive(&recvbuffer[0], 0, header_len, header_len, 0);
											 if (recv_len != header_len)
											 {
												 sslClientUplink.stopWork();
												 sslClientDownlink.stopWork();
												 PRINT_ERROR("receive head from server failed");
												 return;
											 }
											 else
											 {
												 PRINT_INFO("receive head from server");
												 if (rcvHeader.parseBuffer(&recvbuffer[0], 0, header_len) < 0)
												 {
													 PRINT_ERROR("receive bad message from server");
													 return;
												 }
												 uint8_t dataType;
												 getValue(rcvHeader.dataType, dataType);
												 PRINT_INFO("receive message %x from server", dataType);
												 int rcv_msg_len;
												 getValue(rcvHeader.remainLength, rcv_msg_len);
												 recvbuffer.resize(rcv_msg_len);
												 int recv_len = sslClientUplink.receive(&recvbuffer[0], 0, rcv_msg_len, rcv_msg_len, recv_timeout_second);
												 if (recv_len != rcv_msg_len)
												 {
													 sslClientUplink.stopWork();
													 sslClientDownlink.stopWork();
													 PRINT_ERROR("receive remaining from server failed");
													 return;
												 }
												 PRINT_INFO("receive remaining %d from server", rcv_msg_len);
											 }
										 }
									 });
	th_VEH2CLOUD_RUN_sender.join();
	th_response_receiver.join();
	return true;
}
