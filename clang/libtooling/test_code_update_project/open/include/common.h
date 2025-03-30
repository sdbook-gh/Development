#pragma once
#include <typeinfo>
#include <string>

enum NMOpenEnumOpen {
    INFO = 0,
    WARN = 1,
    ERROR = 2,
    FATAL = 3,
    DEBUG = 4,
    TRACE = 5,
    NONE = 6,
    UNKNOWN = 7,
    NM_LOG_ALL = 8,
};

enum NM_LOG_Enum_Common {
    NM_LOG_COMMON_PREFIX = 1,
};

typedef enum {
  NM_LOG_EXTRA_PREFIX = 1,
} NM_LOG_Enum_Extra;

namespace OPEN_NM {
template <typename T>
std::string NM_to_string() {
    return typeid(T).name();
}
}
