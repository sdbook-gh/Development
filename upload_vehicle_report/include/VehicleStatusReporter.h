#ifndef CPPSSLCLIENT__VEHICLESTATUSREPORTER_H
#define CPPSSLCLIENT__VEHICLESTATUSREPORTER_H

#include "SSLClient.h"

namespace v2x
{
	class VehicleStatusReporter
	{
	public:
		bool startReportVehicleStatus(v2x::SSLClient &sslClientUplink, v2x::SSLClient &sslClientDownlink);
	};
} // namespace v2x
#endif //CPPSSLCLIENT__VEHICLESTATUSREPORTER_H
