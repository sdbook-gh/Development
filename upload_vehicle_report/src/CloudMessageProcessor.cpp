#include "CloudMessageProcessor.h"
#include "cloud-msg/MessageDefinition.h"
#include "Utils.h"
#include <vector>
#include <cstdint>
#include <thread>
#include <chrono>
#include <sys/prctl.h>
#include "ShareResource.h"

using namespace v2x;
#define VEH2CLOUD_REQ_SEND_COUNT 1
#define SLEEP_MS 10000

bool
CloudMessageProcessor::HandleServerMessage(v2x::SSLClient &sslClientDownlink, v2x::SSLClient &sslClientUplink)
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

	std::thread th_VEH2CLOUD_REQ_Sender([&sslClientDownlink, &sslClientUplink, vehicleId, send_timeout_second]() -> void
										{
											set_thread_name("th_VEH2CLOUD_REQ_sender");
											PRINT_INFO("th_VEH2CLOUD_REQ_sender started");

											int count = VEH2CLOUD_REQ_SEND_COUNT;
											while (count > 0)
											{
												if (!sslClientDownlink.isWorkable())
												{
													PRINT_ERROR("client status wrong");
													return;
												}
												Header header;
												setValue(header.dataType, DataType::VEH2CLOUD_REQ);
												setValue(header.version, V2XPROTOCOL_VERSION);
												setValue(header.ts_ms, 0);
												setValue(header.ts_min, 0);
												VEH2CLOUD_REQ req(vehicleId);
												setValue(req.ctlMode, 0);
												VEH2CLOUD_REQ_FuncReq req_FuncReq;
												setValue(req_FuncReq.funcReq, 0);
												req.funcReq.push_back(req_FuncReq);
												size_t header_len = header.calcRealSize();
												size_t run_len = req.calcRealSize();
												int total_len = header_len + run_len;
												std::vector<uint8_t> sendbuffer;
												sendbuffer.resize(total_len);
												setValue(header.remainLength, run_len);
												header.fillBuffer(&sendbuffer[0], 0, header_len);
												req.fillBuffer(&sendbuffer[0] + header_len, 0, run_len);
												if (sslClientDownlink.send(&sendbuffer[0], 0, total_len, total_len, send_timeout_second) != total_len)
												{
													sslClientDownlink.stopWork();
													sslClientUplink.stopWork();
													PRINT_ERROR("send VEH2CLOUD_REQ to server failed");
													return;
												}
												else
												{
													PRINT_INFO("send VEH2CLOUD_REQ to server ok");
												}
												std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS));
												count--;
											}
										});

	std::thread th_VEH2CLOUD_REQ_RES_receiver([&sslClientDownlink, &sslClientUplink, recv_timeout_second]() -> void
											  {
												  set_thread_name("th_VEH2CLOUD_REQ_RES_receiver");
												  PRINT_INFO("th_VEH2CLOUD_REQ_RES_receiver started");

												  while (true)
												  {
													  if (!sslClientDownlink.isWorkable())
													  {
														  PRINT_ERROR("client status wrong");
														  return;
													  }
													  std::vector<uint8_t> recvbuffer;
													  Header rcvHeader;
													  int header_len = rcvHeader.calcRealSize();
													  recvbuffer.resize(header_len);
													  // endless wait for response
													  int recv_len = sslClientDownlink.receive(&recvbuffer[0], 0, header_len, header_len, 0);
													  if (recv_len != header_len)
													  {
														  sslClientDownlink.stopWork();
														  sslClientUplink.stopWork();
														  PRINT_ERROR("receive head from server failed");
														  return;
													  }
													  else
													  {
														  if (rcvHeader.parseBuffer(&recvbuffer[0], 0, header_len) < 0)
														  {
															  PRINT_ERROR("receive bad message from server");
															  return;
														  }
														  uint8_t dataType;
														  getValue(rcvHeader.dataType, dataType);
														  PRINT_INFO("receive message %x from server", dataType);
														  if (dataType == v2x::DataType::VEH2CLOUD_REQ_RES)
														  {
															  CLOUD2VEH_REQ_RES reqRes;
															  int reqRes_len;
															  getValue(rcvHeader.remainLength, reqRes_len);
															  recvbuffer.resize(reqRes_len);
															  int recv_len = sslClientDownlink.receive(&recvbuffer[0], 0, reqRes_len, reqRes_len, recv_timeout_second);
															  if (recv_len != reqRes_len)
															  {
																  sslClientDownlink.stopWork();
																  sslClientUplink.stopWork();
																  PRINT_ERROR("receive CLOUD2VEH_REQ_RES from server failed");
																  return;
															  }
															  if (reqRes.parseBuffer(&recvbuffer[0], 0, reqRes_len) < 0)
															  {
																  PRINT_ERROR("receive bad CLOUD2VEH_REQ_RES from server");
																  return;
															  }
															  PRINT_INFO("receive CLOUD2VEH_REQ_RES from server ok");
														  }
														  else if (dataType == v2x::DataType::CLOUD2VEH_CTL)
														  {
															  CLOUD2VEH_CTL ctl("");
															  int ctl_len;
															  getValue(rcvHeader.remainLength, ctl_len);
															  recvbuffer.resize(ctl_len);
															  int recv_len = sslClientDownlink.receive(&recvbuffer[0], 0, ctl_len, ctl_len, recv_timeout_second);
															  if (recv_len != ctl_len)
															  {
																  sslClientDownlink.stopWork();
																  sslClientUplink.stopWork();
																  PRINT_ERROR("receive CLOUD2VEH_CTL from server failed");
																  return;
															  }
															  if (ctl.parseBuffer(&recvbuffer[0], 0, ctl_len) < 0)
															  {
																  PRINT_ERROR("receive bad CLOUD2VEH_CTL from server");
																  return;
															  }
															  PRINT_INFO("receive CLOUD2VEH_CTL from server ok");
														  }
														  else
														  {
															  int rcv_msg_len;
															  getValue(rcvHeader.remainLength, rcv_msg_len);
															  recvbuffer.resize(rcv_msg_len);
															  int recv_len = sslClientDownlink.receive(&recvbuffer[0], 0, rcv_msg_len, rcv_msg_len);
															  if (recv_len != rcv_msg_len)
															  {
																  sslClientDownlink.stopWork();
																  sslClientUplink.stopWork();
																  PRINT_ERROR("receive remaining from server failed");
																  return;
															  }
															  PRINT_INFO("receive remaining %d from server", rcv_msg_len);
														  }
													  }
												  }
											  });
	th_VEH2CLOUD_REQ_Sender.join();
	th_VEH2CLOUD_REQ_RES_receiver.join();
	return true;
}
