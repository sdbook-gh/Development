#include "stack_trace.h"

namespace open {
namespace nmutils {
namespace nmtrace {

std::vector<std::string> NMStackTrace::GetNMStackTrace() const {
  std::vector<std::string> NMStackTrace;

  unw_cursor_t cursor;
  unw_context_t context;
  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  while (unw_step(&cursor) > 0) {
    unw_word_t offset;
    char symbol[256];
    unw_get_proc_name(&cursor, symbol, sizeof(symbol), &offset);
    NMStackTrace.push_back(symbol);
  }

  return NMStackTrace;
}

} // namespace nmtrace
} // namespace nmutils
} // namespace open
