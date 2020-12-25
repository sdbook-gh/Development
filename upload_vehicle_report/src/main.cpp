#include "SSLClient.h"
#include "CloudMessageProcessor.h"
#include "Utils.h"
#include <thread>
#include <chrono>
#include <memory>
#include <exception>
#include "CloudMessageProcessor.h"
#include "VehicleStatusReporter.h"
#include "VehicleReportReceiver.h"
#include "ShareResource.h"

using namespace v2x;

SSLClient sslClientUplink("uplink");
SSLClient sslClientDownlink("downlink");

std::atomic<bool> sslClientUplinkOK(false);
std::atomic<bool> sslClientDownlinkOK(false);
std::atomic<bool> receiverStopped(false);

//#define TEST_SINGLE

void
initUplinkClient()
{
	set_thread_name("initUplinkClient");

	int connect_retry_interval;
	{
		std::unique_lock<std::mutex> lock(ShareResource::configMutex);
		connect_retry_interval = ShareResource::config.getYAMLConfigItem(GLOBAL)["connect_server_retry_interval"].as<int>();
	}
	while (true)
	{
		try
		{
			bool stoppedtrue = true;
			if (receiverStopped.compare_exchange_strong(stoppedtrue, true))
			{
				PRINT_INFO("uplink stopped");
				return;
			}
			sslClientUplink.reset();
			bool result;
			result = sslClientUplink.connectToServer();
			if (result)
			{
				result = sslClientUplink.init();
				if (result)
				{
					result = sslClientUplink.handShake();
					if (result)
					{
						sslClientUplink.displayServerCertificate();
						sslClientUplinkOK = true;
						#ifndef TEST_SINGLE
						bool sslClientDownlinkFalse = false;
						while (sslClientDownlinkOK.compare_exchange_strong(sslClientDownlinkFalse, false))
						{
							std::this_thread::sleep_for(std::chrono::seconds(1));
						}
						#endif
						VehicleStatusReporter reporter;
						reporter.startReportVehicleStatus(sslClientUplink, sslClientDownlink);
					}
				}
				sslClientUplink.closeConnection();
			}
			std::this_thread::sleep_for(std::chrono::seconds(connect_retry_interval));
		}
		catch (const std::exception &e)
		{
			PRINT_ERROR("exception:%s", e.what());
		}
		catch (...)
		{
			PRINT_ERROR("unknown exception got");
		}
	}
}

void
initDownLinkClient()
{
	set_thread_name("initDownLinkClient");

	int connect_retry_interval;
	{
		std::unique_lock<std::mutex> lock(ShareResource::configMutex);
		connect_retry_interval = ShareResource::config.getYAMLConfigItem(GLOBAL)["connect_server_retry_interval"].as<int>();
	}
	while (true)
	{
		try
		{
			bool stoppedtrue = true;
			if (receiverStopped.compare_exchange_strong(stoppedtrue, true))
			{
				PRINT_INFO("uplink stopped");
				return;
			}
			sslClientDownlink.reset();
			bool result;
			result = sslClientDownlink.connectToServer();
			if (result)
			{
				result = sslClientDownlink.init();
				if (result)
				{
					result = sslClientDownlink.handShake();
					if (result)
					{
						sslClientDownlink.displayServerCertificate();
						sslClientDownlinkOK = true;
						#ifndef TEST_SINGLE
						bool sslClientUplinkFalse = false;
						while (sslClientUplinkOK.compare_exchange_strong(sslClientUplinkFalse, false))
						{
							std::this_thread::sleep_for(std::chrono::seconds(1));
						}
						#endif
						CloudMessageProcessor processor;
						processor.HandleServerMessage(sslClientDownlink, sslClientUplink);
					}
				}
				sslClientDownlink.closeConnection();
			}
			std::this_thread::sleep_for(std::chrono::seconds(connect_retry_interval));
		}
		catch (const std::exception &e)
		{
			PRINT_ERROR("exception:%s", e.what());
		}
		catch (...)
		{
			PRINT_ERROR("unknown exception got");
		}
	}
}

void
initVehicleReportReceiver(const int argc, const char *const argv[])
{
	set_thread_name("initVehicleReportReceiver");

	try
	{
		rclcpp::init(argc, argv);
		std::shared_ptr<VehicleReportReceiver> uploader = std::make_shared<VehicleReportReceiver>("VehicleReportReceiver");
		rclcpp::spin(uploader);
	}
	catch (const std::exception &e)
	{
		PRINT_ERROR("exception:%s", e.what());
	}
	catch (...)
	{
		PRINT_ERROR("unknown exception got");
	}
	receiverStopped = true;
	sslClientUplink.stopWork();
	sslClientUplinkOK = false;
	sslClientDownlink.stopWork();
	sslClientDownlinkOK = false;
	rclcpp::shutdown();
}

int
main(const int argc, const char *const argv[])
{
	set_thread_name("main");

	std::thread th1(initUplinkClient);
	std::thread th2(initDownLinkClient);
	std::thread th3(initVehicleReportReceiver, argc, argv);
	th1.join();
	th2.join();
	th3.join();
}
