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
    ALL = 8,
};

namespace NM {
template <typename T>
std::string NM_to_string() {
    return typeid(T).name();
}
}
