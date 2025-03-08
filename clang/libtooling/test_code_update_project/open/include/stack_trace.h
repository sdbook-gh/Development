#pragma once
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <string>
#include <vector>

namespace open {
namespace nmutils {
namespace nmtrace {

class NMStackTrace {
public:
  std::vector<std::string> GetNMStackTrace() const;
};

} // namespace nmtrace
} // namespace nmutils
} // namespace open
