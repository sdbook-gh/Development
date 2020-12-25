#ifndef CPPSSLCLIENT_CONSTANT_H
#define CPPSSLCLIENT_CONSTANT_H

#include <cstdint>

namespace v2x
{

	static constexpr uint8_t V2XHEAD_TYPE = 0xF2;
	static constexpr uint8_t V2XPROTOCOL_VERSION = 0x02;

	struct DataType
	{
		static constexpr uint8_t VEH2CLOUD_RUN = 0x15;
		static constexpr uint8_t VEH2CLOUD_REQ = 0x36;
		static constexpr uint8_t VEH2CLOUD_REQ_RES = 0x37;
		static constexpr uint8_t CLOUD2VEH_CTL = 0x1E;
	};

} // namespace v2x

#endif //CPPSSLCLIENT_CONSTANT_H
