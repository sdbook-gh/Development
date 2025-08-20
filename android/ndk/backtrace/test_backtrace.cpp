#include <backtrace.h>
#include <backtrace-supported.h>

#include <cxxabi.h> // 用于__cxa_demangle
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int g_crash_or_exception = 0;

// 测试代码
class TestClass {
private:
  int* dangerous_ptr;

public:
  TestClass() : dangerous_ptr(nullptr) {}

  void method_level_3() {
    std::cout << "About to crash in method_level_3..." << std::endl;
    if (g_crash_or_exception == 0) {
      *dangerous_ptr = 42;
    } else {
      throw std::runtime_error("Test exception");
    }
  }

  void method_level_2() {
    std::cout << "In method_level_2" << std::endl;
    method_level_3();
  }

  void method_level_1() {
    std::cout << "In method_level_1" << std::endl;
    method_level_2();
  }

  template <typename T>
  void template_method(T value) {
    std::cout << "Template method with value: " << value << std::endl;
    method_level_1();
  }
};

void global_function_c() {
  TestClass obj;
  obj.template_method(123);
}

void global_function_b() {
  std::cout << "In global_function_b" << std::endl;
  global_function_c();
}

void global_function_a() {
  std::cout << "In global_function_a" << std::endl;
  global_function_b();
}

template <int parameter>
void test_lambda_crash() {
  auto lambda = [&](int x) {
    std::cout << "template parameter " << parameter << " Lambda with parameter: " << x << std::endl;
    global_function_a();
  };
  lambda(456);
}

// 全局Backtrace状态
static struct backtrace_state* g_backtrace_state = nullptr;
// 错误回调
static void error_callback(void* data, const char* msg, int errnum) { printf("libbacktrace ERROR: %s (code: %d)\n", msg, errnum); }
// 符号信息回调
static void syminfo_callback(void* data, uintptr_t pc, const char* symname, uintptr_t symval, uintptr_t symsize) {
  if (symname) {
    printf("    %s [0x%lx]\n", symname, pc);
  } else {
    printf("    ??? [0x%lx]\n", pc);
  }
}

// 完整回溯回调
static int full_callback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function) {
  if (function) {
    printf("    %s at %s:%d [0x%lx]\n", function, filename, lineno, pc);
  } else {
    // 尝试通过地址获取符号
    backtrace_syminfo(g_backtrace_state, pc, syminfo_callback, error_callback, nullptr);
  }
  return 0;
}

int main() {
  // 初始化Backtrace
  g_backtrace_state = backtrace_create_state(nullptr, // 使用当前可执行文件
                                             BACKTRACE_SUPPORTS_THREADS, error_callback, nullptr);

  auto crash_handler = [](int sig) -> void {
    std::cout << "\n*** CRASH DETECTED ***" << std::endl;
    std::cout << "Signal: " << sig << " (" << strsignal(sig) << ")" << std::endl;
    backtrace_full(g_backtrace_state, 0, full_callback, error_callback, nullptr);
    std::cout << "\n*** END CRASH INFO ***" << std::endl;
    std::_Exit(EXIT_FAILURE);
  };
  signal(SIGSEGV, crash_handler);
  signal(SIGABRT, crash_handler);
  signal(SIGFPE, crash_handler);
  signal(SIGILL, crash_handler);
  signal(SIGBUS, crash_handler);
  auto handle_terminate = []() -> void {
    backtrace_full(g_backtrace_state, 0, full_callback, error_callback, nullptr);
    std::_Exit(EXIT_FAILURE);
  };
  std::set_terminate(handle_terminate);
  g_crash_or_exception = 0;
  test_lambda_crash<0>();
  return 0;
}
