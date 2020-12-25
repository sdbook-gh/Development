#ifndef CPPSSLCLIENT_MESSAGEDEFINITION_H
#define CPPSSLCLIENT_MESSAGEDEFINITION_H

#include "Constant.h"
#include <math.h>
#include <cstdint>
#include <vector>
#include <type_traits>
#include "../Utils.h"
#include <endian.h>
#include <typeinfo>
#include <cstring>
#include <memory>
#include <vector>

#pragma pack(push, 1)

namespace v2x
{

struct BYTE
{
	uint8_t value;
};
struct WORD
{
	uint16_t value;
};
struct INT
{
	int32_t value;
};
struct UINT
{
	uint32_t value;
};
struct LONG
{
	int64_t value;
};
struct ULONG
{
	uint64_t value;
};
struct FLOAT
{
	float_t value;
};
struct DOUBLE
{
	double_t value;
};
struct TS_MIN
{
	uint32_t value;
};
struct TS_MS
{
	uint16_t value;
};
struct TIMESTAMP
{
	uint64_t value;
};
struct STRING
{
	char value;
};

template<typename DataType>
size_t
parseFrom(const uint8_t *const buffer, const size_t offset, DataType &value)
{
	if (std::is_same<typename std::decay<DataType>::type, v2x::BYTE>::value)
	{
		const uint8_t *const ptr = (const uint8_t *)(buffer + offset);
		*(uint8_t *)&value = *ptr;
		return sizeof(uint8_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::WORD>::value)
	{
		const uint16_t *const ptr = (const uint16_t *)(buffer + offset);
		*(uint16_t *)&value = be16toh(*ptr);
		return sizeof(uint16_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::INT>::value)
	{
		const int32_t *const ptr = (const int32_t *)(buffer + offset);
		*(int32_t *)&value = be32toh(*ptr);
		return sizeof(int32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::UINT>::value)
	{
		const uint32_t *const ptr = (const uint32_t *)(buffer + offset);
		*(uint32_t *)&value = be32toh(*ptr);
		return sizeof(uint32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::LONG>::value)
	{
		const int64_t *const ptr = (const int64_t *)(buffer + offset);
		*(int64_t *)&value = be64toh(*ptr);
		return sizeof(int64_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::ULONG>::value)
	{
		const uint64_t *const ptr = (const uint64_t *)(buffer + offset);
		*(uint64_t *)&value = be64toh(*ptr);
		return sizeof(uint64_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::FLOAT>::value)
	{
		const uint32_t *const ptr = (const uint32_t *)(buffer + offset);
		*(uint32_t *)&value = (float)be32toh(*ptr);
		return sizeof(uint32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::DOUBLE>::value)
	{
		const uint64_t *const ptr = (const uint64_t *)(buffer + offset);
		*(uint64_t *)&value = (double)be64toh(*ptr);
		return sizeof(uint64_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::TS_MIN>::value)
	{
		const uint32_t *const ptr = (const uint32_t *)(buffer + offset);
		*(uint32_t *)&value = be32toh(*ptr);
		return sizeof(uint32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::TS_MS>::value)
	{
		const uint16_t *const ptr = (const uint16_t *)(buffer + offset);
		*(uint16_t *)&value = be16toh(*ptr);
		return sizeof(uint16_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::TIMESTAMP>::value)
	{
		const uint64_t *const ptr = (const uint64_t *)(buffer + offset);
		*(uint64_t *)&value = be64toh(*ptr);
		return sizeof(uint64_t);
	}
	PRINT_ERROR("unsupported parse type");
	exit(ERROR_PARSE);
}

template<typename DataType, size_t size>
size_t
parseFrom(const uint8_t *const buffer, const size_t offset, DataType (&value)[size])
{
	if (std::is_same<typename std::decay<DataType>::type, v2x::STRING>::value)
	{
		const uint8_t *const srcptr = (uint8_t *)(buffer + offset);
		uint8_t *const destptr = (uint8_t *)value;
		bcopy(srcptr, destptr, size);
		return sizeof(uint8_t) * size;
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::BYTE>::value)
	{
		const uint8_t *const srcptr = (uint8_t *)(buffer + offset);
		uint8_t *const destptr = (uint8_t *)&value;
		for (size_t i = 0; i < size; i++)
		{
			destptr[size - i - 1] = srcptr[i];
		}
		return sizeof(uint8_t) * size;
	}
	PRINT_ERROR("unsupported parse type");
	exit(ERROR_PARSE);
}

template<typename DataType>
size_t
fillTo(uint8_t *const buffer, const size_t offset, const DataType &value)
{
	if (std::is_same<typename std::decay<DataType>::type, v2x::BYTE>::value)
	{
		uint8_t *const ptr = (uint8_t *)(buffer + offset);
		*ptr = *(uint8_t *)&value;
		return sizeof(uint8_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::WORD>::value)
	{
		uint16_t *const ptr = (uint16_t *)(buffer + offset);
		*ptr = htobe16(*(uint16_t *)&value);
		return sizeof(uint16_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::INT>::value)
	{
		int32_t *const ptr = (int32_t *)(buffer + offset);
		*ptr = htobe32(*(int32_t *)&value);
		return sizeof(int32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::UINT>::value)
	{
		uint32_t *const ptr = (uint32_t *)(buffer + offset);
		*ptr = htobe32(*(uint32_t *)&value);
		return sizeof(uint32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::LONG>::value)
	{
		int64_t *const ptr = (int64_t *)(buffer + offset);
		*ptr = htobe64(*(int64_t *)&value);
		return sizeof(int64_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::ULONG>::value)
	{
		uint64_t *const ptr = (uint64_t *)(buffer + offset);
		*ptr = htobe64(*(uint64_t *)&value);
		return sizeof(uint64_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::FLOAT>::value)
	{
		uint32_t *const ptr = (uint32_t *)(buffer + offset);
		*ptr = htobe32(*(uint32_t *)&value);
		return sizeof(uint32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::DOUBLE>::value)
	{
		uint64_t *const ptr = (uint64_t *)(buffer + offset);
		*ptr = htobe64(*(uint64_t *)&value);
		return sizeof(uint64_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::TS_MIN>::value)
	{
		uint32_t *const ptr = (uint32_t *)(buffer + offset);
		*ptr = htobe32(*(uint32_t *)&value);
		return sizeof(uint32_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::TS_MS>::value)
	{
		uint16_t *const ptr = (uint16_t *)(buffer + offset);
		*ptr = htobe16(*(uint16_t *)&value);
		return sizeof(uint16_t);
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::TIMESTAMP>::value)
	{
		uint64_t *const ptr = (uint64_t *)(buffer + offset);
		*ptr = htobe64(*(uint64_t *)&value);
		return sizeof(uint64_t);
	}
	PRINT_ERROR("unsupported fill type");
	exit(ERROR_FILL);
}

template<typename DataType, size_t size>
size_t
fillTo(uint8_t *const buffer, const size_t offset, const DataType (&value)[size])
{
	if (std::is_same<typename std::decay<DataType>::type, v2x::STRING>::value)
	{
		uint8_t *const destptr = (uint8_t *)(buffer + offset);
		const uint8_t *const srcptr = (const uint8_t *)value;
		bcopy(srcptr, destptr, size);
		return sizeof(uint8_t) * size;
	}
	else if (std::is_same<typename std::decay<DataType>::type, v2x::BYTE>::value)
	{
		uint8_t *const destptr = (uint8_t *)(buffer + offset);
		const uint8_t *const srcptr = (const uint8_t *)&value;
		for (size_t i = 0; i < size; i++)
		{
			destptr[size - i - 1] = srcptr[i];
		}
		return sizeof(uint8_t) * size;
	}
	PRINT_ERROR("unsupported fill type");
	exit(ERROR_FILL);
}

template<typename DataType, typename InputType>
void
setValue(DataType &value, const InputType input)
{
	if (sizeof(DataType) >= sizeof(InputType))
	{
		bcopy(&input, &value, sizeof(InputType));
	}
	else
	{
		bcopy(&input, (uint8_t *)&value, sizeof(DataType));
	}
	if (sizeof(DataType) > sizeof(InputType))
	{
		bzero(((uint8_t *)&value + sizeof(InputType)), sizeof(DataType) - sizeof(InputType));
	}
}

template<typename DataType, typename OutputType>
void
getValue(const DataType &value, OutputType &output)
{
	if (sizeof(DataType) > sizeof(OutputType))
	{
		PRINT_ERROR("output too short to getValue");
		exit(ERROR_GETVALUE);
	}
	bcopy(&value, &output, sizeof(DataType));
	if (sizeof(OutputType) > sizeof(DataType))
	{
		bzero(((uint8_t *)&output + sizeof(DataType)), sizeof(OutputType) - sizeof(DataType));
	}
}

template<typename DataType, typename InputType, size_t size>
void
setValue(DataType (&value)[size], const InputType input)
{
	if (std::is_same<typename std::decay<DataType>::type, v2x::BYTE>::value)
	{
		if (sizeof(DataType) <= sizeof(InputType))
		{
			bcopy(&input, &value, sizeof(DataType));
		}
		else
		{
			bcopy(&input, &value, sizeof(InputType));
			bzero(((uint8_t *)&value + sizeof(InputType)), sizeof(DataType) - sizeof(InputType));
		}
		return;
	}
	PRINT_ERROR("unsupported type to setValue");
	exit(ERROR_SETVALUE);
}

template<typename DataType, typename OutputType, size_t size>
void
getValue(const DataType (&value)[size], OutputType &output)
{
	if (std::is_same<typename std::decay<DataType>::type, v2x::BYTE>::value)
	{
		if (sizeof(DataType) > sizeof(OutputType))
		{
			PRINT_ERROR("output too short to getValue");
			exit(ERROR_GETVALUE);
		}
		bcopy(&value, &output, sizeof(DataType));
		if (sizeof(OutputType) > sizeof(DataType))
		{
			bzero(((uint8_t *)&output + sizeof(DataType)), sizeof(OutputType) - sizeof(DataType));
		}
		return;
	}
	PRINT_ERROR("unsupported type to getValue");
	exit(ERROR_GETVALUE);
}

template<size_t size>
void
setValue(v2x::STRING (&value)[size], const std::string input)
{
	if (input.size() - 1 > size)
	{
		PRINT_ERROR("input string too large: %s", input.c_str());
		exit(ERROR_SETVALUE);
	}
	bzero(value, size);
	bcopy(input.c_str(), value, input.size());
}

template<size_t size>
void
getValue(const v2x::STRING (&value)[size], std::string &output)
{
	output = std::string(&value[0], &value[size]);
}

template<typename DataType>
void
setDefault(DataType &value, uint8_t defaultValue = 0xFF)
{
	memset((uint8_t *)&value, defaultValue, sizeof(value));
}

#define CHECK_PARSE_BUFFER(buffer, offset, length)                                                                      \
    if (buffer == nullptr)                                                                                              \
    {                                                                                                                   \
        PRINT_ERROR("buffer is null");                                                                                  \
        return ERROR_PARSE;                                                                                             \
    }                                                                                                                   \
    if (offset + calcRealSize() > length)                                                                               \
    {                                                                                                                   \
        PRINT_ERROR("buffer length is too short: offset %ld calcRealSize %ld length %ld", offset, calcRealSize(), length); \
        return ERROR_PARSE;                                                                                             \
    }

#define CHECK_FILL_BUFFER(buffer, offset, length)                                                                       \
    if (buffer == nullptr)                                                                                              \
    {                                                                                                                   \
        PRINT_ERROR("buffer is null");                                                                                  \
        return ERROR_FILL;                                                                                              \
    }                                                                                                                   \
    if (offset + calcRealSize() > length)                                                                               \
    {                                                                                                                   \
        PRINT_ERROR("buffer length is too short: offset %ld calcRealSize %ld length %ld", offset, calcRealSize(), length); \
        return ERROR_FILL;                                                                                              \
    }

struct Header
{
private:
	v2x::BYTE msgType;

public:
	v2x::BYTE remainLength[3];
	v2x::BYTE dataType;
	v2x::BYTE version;
	v2x::TS_MS ts_ms;
	v2x::TS_MIN ts_min;

	Header()
	{
		setDefault(msgType, 0xF2);
		setDefault(remainLength, 0);
		setDefault(dataType, 0);
		setDefault(version, 0);
		setDefault(ts_ms, 0);
		setDefault(ts_min, 0);
	}

	constexpr inline size_t
	calcRealSize() const
	{
		return sizeof(decltype(*this));
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, msgType);
		pos += parseFrom(buffer, offset + pos, remainLength);
		pos += parseFrom(buffer, offset + pos, dataType);
		pos += parseFrom(buffer, offset + pos, version);
		pos += parseFrom(buffer, offset + pos, ts_ms);
		pos += parseFrom(buffer, offset + pos, ts_min);
		return pos;
	}

	size_t
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, msgType);
		pos += fillTo(buffer, offset + pos, remainLength);
		pos += fillTo(buffer, offset + pos, dataType);
		pos += fillTo(buffer, offset + pos, version);
		pos += fillTo(buffer, offset + pos, ts_ms);
		pos += fillTo(buffer, offset + pos, ts_min);
		return pos;
	}
};

struct VEH2CLOUD_RUN
{
private:
	v2x::STRING vehicleId[8];

public:
	v2x::TIMESTAMP timestampGNSS;
	v2x::WORD velocityGNSS;
	v2x::UINT longitude;
	v2x::UINT latitude;
	v2x::INT elevation;
	v2x::UINT heading;
	v2x::BYTE hdop;
	v2x::BYTE vdop;
	v2x::BYTE tapPos;
	v2x::INT steeringAngle;
	v2x::WORD lights;
	v2x::WORD velocityCAN;
	v2x::WORD acceleration_V;
	v2x::WORD acceleration_H;
	v2x::BYTE accelPos;
	v2x::INT engineSpeed;
	v2x::INT engineTorque;
	v2x::BYTE brakeFlag;
	v2x::BYTE brakePos;
	v2x::WORD brakePressure;
	v2x::WORD yawRate;
	v2x::WORD wheelVelocity_FL;
	v2x::WORD wheelVelocity_FR;
	v2x::WORD wheelVelocity_RL;
	v2x::WORD wheelVelocity_RR;
	v2x::BYTE absFlag;
	v2x::BYTE tcsFlag;
	v2x::BYTE espFlag;
	v2x::BYTE lkaFlag;
	v2x::BYTE accMode;

	VEH2CLOUD_RUN(const std::string inVehicleId)
	{
		setValue(vehicleId, inVehicleId);
		setDefault(timestampGNSS, -1);
		setDefault(velocityGNSS, -1);
		setDefault(longitude, -1);
		setDefault(latitude, -1);
		setDefault(elevation, -1);
		setDefault(heading, -1);
		setDefault(hdop, -1);
		setDefault(vdop, -1);
		setDefault(tapPos, -1);
		setDefault(steeringAngle, -1);
		setDefault(lights, -1);
		setDefault(velocityCAN, -1);
		setDefault(acceleration_V, -1);
		setDefault(acceleration_H, -1);
		setDefault(accelPos, -1);
		setDefault(engineSpeed, -1);
		setDefault(engineTorque, -1);
		setDefault(brakeFlag, -1);
		setDefault(brakePos, -1);
		setDefault(brakePressure, -1);
		setDefault(yawRate, -1);
		setDefault(wheelVelocity_FL, -1);
		setDefault(wheelVelocity_FR, -1);
		setDefault(wheelVelocity_RL, -1);
		setDefault(wheelVelocity_RR, -1);
		setDefault(absFlag, -1);
		setDefault(tcsFlag, -1);
		setDefault(espFlag, -1);
		setDefault(lkaFlag, -1);
		setDefault(accMode, -1);
	}

	constexpr inline size_t
	calcRealSize() const
	{
		return sizeof(decltype(*this));
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, vehicleId);
		pos += parseFrom(buffer, offset + pos, timestampGNSS);
		pos += parseFrom(buffer, offset + pos, velocityGNSS);
		pos += parseFrom(buffer, offset + pos, longitude);
		pos += parseFrom(buffer, offset + pos, latitude);
		pos += parseFrom(buffer, offset + pos, elevation);
		pos += parseFrom(buffer, offset + pos, heading);
		pos += parseFrom(buffer, offset + pos, hdop);
		pos += parseFrom(buffer, offset + pos, vdop);
		pos += parseFrom(buffer, offset + pos, tapPos);
		pos += parseFrom(buffer, offset + pos, steeringAngle);
		pos += parseFrom(buffer, offset + pos, lights);
		pos += parseFrom(buffer, offset + pos, velocityCAN);
		pos += parseFrom(buffer, offset + pos, acceleration_V);
		pos += parseFrom(buffer, offset + pos, acceleration_H);
		pos += parseFrom(buffer, offset + pos, accelPos);
		pos += parseFrom(buffer, offset + pos, engineSpeed);
		pos += parseFrom(buffer, offset + pos, engineTorque);
		pos += parseFrom(buffer, offset + pos, brakeFlag);
		pos += parseFrom(buffer, offset + pos, brakePos);
		pos += parseFrom(buffer, offset + pos, brakePressure);
		pos += parseFrom(buffer, offset + pos, yawRate);
		pos += parseFrom(buffer, offset + pos, wheelVelocity_FL);
		pos += parseFrom(buffer, offset + pos, wheelVelocity_FR);
		pos += parseFrom(buffer, offset + pos, wheelVelocity_RL);
		pos += parseFrom(buffer, offset + pos, wheelVelocity_RR);
		pos += parseFrom(buffer, offset + pos, absFlag);
		pos += parseFrom(buffer, offset + pos, tcsFlag);
		pos += parseFrom(buffer, offset + pos, espFlag);
		pos += parseFrom(buffer, offset + pos, lkaFlag);
		pos += parseFrom(buffer, offset + pos, accMode);
		return pos;
	}

	size_t
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, vehicleId);
		pos += fillTo(buffer, offset + pos, timestampGNSS);
		pos += fillTo(buffer, offset + pos, velocityGNSS);
		pos += fillTo(buffer, offset + pos, longitude);
		pos += fillTo(buffer, offset + pos, latitude);
		pos += fillTo(buffer, offset + pos, elevation);
		pos += fillTo(buffer, offset + pos, heading);
		pos += fillTo(buffer, offset + pos, hdop);
		pos += fillTo(buffer, offset + pos, vdop);
		pos += fillTo(buffer, offset + pos, tapPos);
		pos += fillTo(buffer, offset + pos, steeringAngle);
		pos += fillTo(buffer, offset + pos, lights);
		pos += fillTo(buffer, offset + pos, velocityCAN);
		pos += fillTo(buffer, offset + pos, acceleration_V);
		pos += fillTo(buffer, offset + pos, acceleration_H);
		pos += fillTo(buffer, offset + pos, accelPos);
		pos += fillTo(buffer, offset + pos, engineSpeed);
		pos += fillTo(buffer, offset + pos, engineTorque);
		pos += fillTo(buffer, offset + pos, brakeFlag);
		pos += fillTo(buffer, offset + pos, brakePos);
		pos += fillTo(buffer, offset + pos, brakePressure);
		pos += fillTo(buffer, offset + pos, yawRate);
		pos += fillTo(buffer, offset + pos, wheelVelocity_FL);
		pos += fillTo(buffer, offset + pos, wheelVelocity_FR);
		pos += fillTo(buffer, offset + pos, wheelVelocity_RL);
		pos += fillTo(buffer, offset + pos, wheelVelocity_RR);
		pos += fillTo(buffer, offset + pos, absFlag);
		pos += fillTo(buffer, offset + pos, tcsFlag);
		pos += fillTo(buffer, offset + pos, espFlag);
		pos += fillTo(buffer, offset + pos, lkaFlag);
		pos += fillTo(buffer, offset + pos, accMode);
		return pos;
	}
};

struct VEH2CLOUD_REQ_FuncReq
{
	v2x::BYTE funcReq;

	constexpr inline size_t
	calcRealSize() const
	{
		return sizeof(decltype(*this));
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, funcReq);
		return pos;
	}

	size_t inline
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, funcReq);
		return pos;
	}
};

struct VEH2CLOUD_REQ
{
private:
	v2x::STRING vehicleId[8];

public:
	v2x::BYTE ctlMode;

private:
	v2x::BYTE reqLen;

public:
	std::vector<VEH2CLOUD_REQ_FuncReq> funcReq;

	VEH2CLOUD_REQ(const std::string &inVehicleId)
	{
		setValue(vehicleId, inVehicleId);
		setValue(ctlMode, 0);
		setValue(reqLen, 0);
	}

	inline size_t
	calcRealSize() const
	{
		size_t totalSize = 0;
		if (funcReq.size() > 0)
		{
			totalSize += (funcReq[0].calcRealSize() * funcReq.size());
		}
		totalSize += sizeof(vehicleId) + sizeof(ctlMode) + sizeof(reqLen);
		return totalSize;
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		funcReq.clear();
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, vehicleId);
		pos += parseFrom(buffer, offset + pos, ctlMode);
		pos += parseFrom(buffer, offset + pos, reqLen);
		size_t reqLenValue;
		getValue(reqLen, reqLenValue);
		for (size_t i = 0; i < reqLenValue; i++)
		{
			VEH2CLOUD_REQ_FuncReq funcReqItem;
			int ret = funcReqItem.parseBuffer(buffer, pos, length);
			if (ret < 0)
				return ERROR_PARSE;
			pos += ret;
			funcReq.push_back(funcReqItem);
		}
		return pos;
	}

	size_t
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, vehicleId);
		pos += fillTo(buffer, offset + pos, ctlMode);
		setValue(reqLen, funcReq.size());
		pos += fillTo(buffer, offset + pos, reqLen);
		for (size_t i = 0; i < funcReq.size(); i++)
		{
			pos += funcReq[i].fillBuffer(buffer, offset + pos, length);
		}
		return pos;
	}
};

struct CLOUD2VEH_REQ_RES_FuncRes
{
	v2x::BYTE funcRes;

	constexpr inline size_t
	calcRealSize() const
	{
		return sizeof(decltype(*this));
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, funcRes);
		return pos;
	}

	size_t inline
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, funcRes);
		return pos;
	}
};

struct CLOUD2VEH_REQ_RES
{
	std::vector<CLOUD2VEH_REQ_RES_FuncRes> funcRes;

	inline size_t
	calcRealSize() const
	{
		size_t totalSize = 0;
		if (funcRes.size() > 0)
		{
			totalSize += (funcRes[0].calcRealSize() * funcRes.size());
		}
		totalSize += sizeof(resLen);
		return totalSize;
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		funcRes.clear();
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, resLen);
		size_t resLenValue;
		getValue(resLen, resLenValue);
		for (size_t i = 0; i < resLenValue; i++)
		{
			CLOUD2VEH_REQ_RES_FuncRes funcResItem;
			int ret = funcResItem.parseBuffer(buffer, pos, length);
			if (ret < 0)
			{
				return ERROR_PARSE;
			}
			funcRes.push_back(funcResItem);
		}
		return pos;
	}

	size_t
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		setValue(resLen, funcRes.size());
		pos += fillTo(buffer, offset + pos, resLen);
		for (size_t i = 0; i < funcRes.size(); i++)
		{
			pos += funcRes[i].fillBuffer(buffer, offset + pos, length);
		}
		return pos;
	}

private:
	v2x::BYTE resLen;
};

struct CLOUD2VEH_CTL_CtlData_Equation
{
	v2x::DOUBLE factor3;
	v2x::DOUBLE factor2;
	v2x::DOUBLE factor1;
	v2x::DOUBLE factorC;
	v2x::WORD min;
	v2x::WORD max;

	constexpr inline size_t
	calcRealSize() const
	{
		return sizeof(decltype(*this));
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, factor3);
		pos += parseFrom(buffer, offset + pos, factor2);
		pos += parseFrom(buffer, offset + pos, factor1);
		pos += parseFrom(buffer, offset + pos, factorC);
		pos += parseFrom(buffer, offset + pos, min);
		pos += parseFrom(buffer, offset + pos, max);
		return pos;
	}

	size_t inline
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, factor3);
		pos += fillTo(buffer, offset + pos, factor2);
		pos += fillTo(buffer, offset + pos, factor1);
		pos += fillTo(buffer, offset + pos, factorC);
		pos += fillTo(buffer, offset + pos, min);
		pos += fillTo(buffer, offset + pos, max);
		return pos;
	}
};

struct CLOUD2VEH_CTL_CtlData
{
public:
	v2x::WORD expSpeed;

private:
	v2x::BYTE equationNum;

public:
	std::vector<CLOUD2VEH_CTL_CtlData_Equation> equation;

	CLOUD2VEH_CTL_CtlData()
	{
		setValue(expSpeed, 0);
		setValue(equationNum, 0);
	}

	inline size_t
	calcRealSize() const
	{
		size_t totalSize = 0;
		if (equation.size() > 0)
		{
			totalSize += (equation[0].calcRealSize() * equation.size());
		}
		totalSize += sizeof(expSpeed) + sizeof(equationNum);
		return totalSize;
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		equation.clear();
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, expSpeed);
		pos += parseFrom(buffer, offset + pos, equationNum);
		size_t equationNumValue;
		getValue(equationNum, equationNumValue);
		for (size_t i = 0; i < equationNumValue; i++)
		{
			CLOUD2VEH_CTL_CtlData_Equation equationItem;
			int ret = equationItem.parseBuffer(buffer, pos, length);
			if (ret < 0)
				return ERROR_PARSE;
			pos += ret;
			equation.push_back(equationItem);
		}
		return pos;
	}

	size_t
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, expSpeed);
		setValue(equationNum, equation.size());
		pos += fillTo(buffer, offset + pos, equationNum);
		for (size_t i = 0; i < equation.size(); i++)
		{
			pos += equation[i].fillBuffer(buffer, offset + pos, length);
		}
		return pos;
	}
};

struct CLOUD2VEH_CTL
{
private:
	v2x::STRING vehicleId[8];

public:
	v2x::BYTE ctlMode;

private:
	v2x::BYTE dataLen;

public:
	std::vector<CLOUD2VEH_CTL_CtlData> ctlData;

	CLOUD2VEH_CTL(const std::string &inVehicleId)
	{
		setValue(vehicleId, inVehicleId);
		setValue(ctlMode, 0);
		setValue(dataLen, 0);
	}

	inline size_t
	calcRealSize() const
	{
		size_t totalSize = 0;
		if (ctlData.size() > 0)
		{
			totalSize += (ctlData[0].calcRealSize() * ctlData.size());
		}
		totalSize += sizeof(vehicleId) + sizeof(ctlMode) + sizeof(dataLen);
		return totalSize;
	}

	inline int
	parseBuffer(const uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_PARSE_BUFFER(buffer, offset, length)
		ctlData.clear();
		int pos = 0;
		pos += parseFrom(buffer, offset + pos, vehicleId);
		pos += parseFrom(buffer, offset + pos, ctlMode);
		pos += parseFrom(buffer, offset + pos, dataLen);
		size_t dataLenValue;
		getValue(dataLen, dataLenValue);
		for (size_t i = 0; i < dataLenValue; i++)
		{
			CLOUD2VEH_CTL_CtlData ctlDataItem;
			int ret = ctlDataItem.parseBuffer(buffer, pos, length);
			if (ret < 0)
				return ERROR_PARSE;
			pos += ret;
			ctlData.push_back(ctlDataItem);
		}
		return pos;
	}

	size_t
	fillBuffer(uint8_t *const buffer, const size_t offset, const size_t length)
	{
		CHECK_FILL_BUFFER(buffer, offset, length)
		int pos = 0;
		pos += fillTo(buffer, offset + pos, vehicleId);
		pos += fillTo(buffer, offset + pos, ctlMode);
		setValue(dataLen, ctlData.size());
		pos += fillTo(buffer, offset + pos, dataLen);
		for (size_t i = 0; i < ctlData.size(); i++)
		{
			pos += ctlData[i].fillBuffer(buffer, offset + pos, length);
		}
		return pos;
	}
};

} // namespace v2x

#pragma pack(pop)

#endif //CPPSSLCLIENT_MESSAGEDEFINITION_H
