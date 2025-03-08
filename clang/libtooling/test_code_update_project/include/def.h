#pragma once

#define BEGIN_NS namespace NM {
#define END_NS }
// #define NSDEF NM
#define NS_PREFIX(DECL) NM##DECL

BEGIN_NS

namespace nm_values {

typedef enum {
  NS_PREFIX(ENUM_1) = 1,
  NS_PREFIX(ENUM_2) = 2,
  NS_PREFIX(ENUM_3) = 3,
} NS_PREFIX(TestEnum1);

enum class NMTestEnum2 {
  NM_ENUM_4 = 4,
  NM_ENUM_5 = 5,
  NM_ENUM_6 = 6,
};

enum NS_PREFIX(OpenEnum) {
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

} // namespace nm_values

END_NS
