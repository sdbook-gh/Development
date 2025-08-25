#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <unwind.h>
#include <dlfcn.h>
#include <cstdint>

void print_stacktrace_backtrace() {
  void *buffer[100];
  int nptrs = backtrace(buffer, 100);
  char **strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < nptrs; i++) { printf("%s\n", strings[i]); }
  free(strings);
}

static _Unwind_Reason_Code trace(struct _Unwind_Context *ctx, void *arg) {
  uintptr_t pc = _Unwind_GetIP(ctx);
  if (!pc) return _URC_END_OF_STACK;
  --pc;
  // https://www.kimi.com/share/d2m0oi5nfo2tf2no4or0
  Dl_info info;
  if (dladdr((void *)pc, &info) && info.dli_fname) {
    size_t offset = pc - (uintptr_t)info.dli_fbase;
    printf("%s 0x%p +0x%zx\n", info.dli_fname, info.dli_fbase, offset);
  }
  return _URC_NO_REASON;
}

void print_backtrace(void) { _Unwind_Backtrace(trace, nullptr); }

void bar(void) {
  print_backtrace();
  printf("======\n");
  print_stacktrace_backtrace();
}

void foo(void) { bar(); }

#include <iostream>
int main(void) {
  foo();
  char val;
  std::cin >> val;
  return 0;
}
