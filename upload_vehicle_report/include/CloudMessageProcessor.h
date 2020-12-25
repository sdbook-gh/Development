#ifndef CPPSSLCLIENT_CLOUDMESSAGEPROCESSOR_H
#define CPPSSLCLIENT_CLOUDMESSAGEPROCESSOR_H

#include "SSLClient.h"

namespace v2x
{
	class CloudMessageProcessor
	{
	public:
		bool HandleServerMessage(v2x::SSLClient &sslClientDownlink, v2x::SSLClient &sslClientUplink);
	};
} // namespace v2x

#endif //CPPSSLCLIENT_CLOUDMESSAGEPROCESSOR_H
