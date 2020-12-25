#ifndef CPPSSLCLIENT_UTILS_H
#define CPPSSLCLIENT_UTILS_H

//#define ENABLE_STACK_TRACE

#include <cstdio>
#include <sstream>
#include <thread>
#include <mutex>
#ifdef ENABLE_STACK_TRACE
#include "backward.hpp"
#endif
#include <sstream>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <rclcpp/rclcpp.hpp>
#include "yaml-cpp/yaml.h"
#include <functional>

namespace v2x
{

enum ErrorType
{
	ERROR_UNKNOWN = -1,
	ERROR_CONFIG = -11,
	ERROR_CONFIG_ITEM = -12,
	ERROR_SSL = -21,
	ERROR_NETWORK = -31,
	ERROR_NETWORK_EPOLL = -32,
	ERROR_NETWORK_TIMEOUT = -33,
	ERROR_PARSE = -41,
	ERROR_FILL = -42,
	ERROR_SETVALUE = -43,
	ERROR_GETVALUE = -44,
	ERROR_STATUS_STOPPED = -51,
};

const char *const GLOBAL = "global";
const char *const VEHICLE = "vehicle";

inline void
set_thread_name(const char *const);
inline const char *
get_thread_name();

class ThreadUtil
{
private:
	static thread_local std::string current_thread_name;
	friend inline void
	set_thread_name(const char *const);
	friend inline const char *
	get_thread_name();
};

inline void
set_thread_name(const char *const th_name)
{
	ThreadUtil::current_thread_name = th_name;
}

inline const char *
get_thread_name()
{
	return ThreadUtil::current_thread_name.c_str();
}

#define PRINT_INFO(fmt, ...) \
    { RCUTILS_LOG_INFO("%s ---- %s " fmt, get_thread_name(), __PRETTY_FUNCTION__, ##__VA_ARGS__); }

#ifdef ENABLE_STACK_TRACE
#define PRINT_STACK_TRACE \
    {                      \
    backward::StackTrace st; \
    st.load_here(32); \
    backward::Printer p; \
    std::stringstream stream; \
    p.print(st, stream); \
    RCUTILS_LOG_ERROR("%s", stream.str().c_str()); \
    }
#else
#define PRINT_STACK_TRACE
#endif

#define PRINT_ERROR(fmt, ...) \
    { \
    RCUTILS_LOG_ERROR("%s ---- %s_%d " fmt, get_thread_name(), __PRETTY_FUNCTION__, __LINE__, ##__VA_ARGS__); \
    PRINT_STACK_TRACE \
    }

template<typename... Args>
void
print_ssl_error(const char *const file, const char *const function, const int line, const char *const reason)
{
	thread_local BIO *bio = BIO_new(BIO_s_mem());
	ERR_print_errors(bio);
	char *buf;
	size_t len = BIO_get_mem_data(bio, &buf);
	std::string ssl_error_str(buf, len);
	RCUTILS_LOG_ERROR("%s ---- %s %s %d %s %s", get_thread_name(), file, function, line, reason, ssl_error_str.c_str());
}

#define PRINT_SSL_ERROR(prefix)  v2x::print_ssl_error(__FILE__, __PRETTY_FUNCTION__, __LINE__, prefix)

class Config
{
public:
	Config();
	YAML::Node
	getYAMLConfigItem(const std::string &nodeName);
private:
	YAML::Node cfgNode;
};

class Guard
{
private:
	std::function<void()> destruct_function;
public:
	template<typename T>
	Guard(T &&construct_function, std::function<void()> &&d_function) : destruct_function(std::forward<std::function<void()>>(d_function))
	{
		construct_function();
	}
	~Guard()
	{
		destruct_function();
	}
};

} // namespace v2x

#endif //CPPSSLCLIENT_UTILS_H
