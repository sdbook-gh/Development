// #include <iostream>
// #include <string>

// // libunwind - for stack unwinding, sudo apt install libunwind
// #define UNW_LOCAL_ONLY
// #include <libunwind.h>

// // elfutils/libdw - for DWARF parsing
// #include <cxxabi.h>
// #include <dwarf.h>
// #include <elfutils/libdwfl.h>
// #include <signal.h>
// #include <unistd.h> // for getpid()

// #include <cstring>

// // A helper class to manage libdwfl resources
// class DwarfResolver {
// public:
//   DwarfResolver() {
//     char* debuginfo_path = nullptr;
//     // The callbacks structure for dwfl_begin
//     static const Dwfl_Callbacks callbacks = {
//       .find_elf = dwfl_linux_proc_find_elf,
//       .find_debuginfo = dwfl_standard_find_debuginfo,
//       .debuginfo_path = &debuginfo_path,
//     };

//     // Initialize Dwfl for the current process
//     dwfl = dwfl_begin(&callbacks);
//     if (!dwfl) {
//       std::cerr << "dwfl_begin failed" << std::endl;
//       return;
//     }

//     // Report information about the current process
//     if (dwfl_linux_proc_report(dwfl, getpid()) != 0) {
//       std::cerr << "dwfl_linux_proc_report failed: " << dwfl_errmsg(-1) << std::endl;
//       dwfl_end(dwfl);
//       dwfl = nullptr;
//     }

//     // Finalize the report
//     if (dwfl_report_end(dwfl, nullptr, nullptr) != 0) {
//       std::cerr << "dwfl_report_end failed: " << dwfl_errmsg(-1) << std::endl;
//       dwfl_end(dwfl);
//       dwfl = nullptr;
//     }
//   }

//   ~DwarfResolver() {
//     if (dwfl) { dwfl_end(dwfl); }
//   }

//   // Resolves a program counter to file, line, and function name
//   bool resolve_address(uintptr_t pc, std::string& file, int& line, std::string& func) {
//     if (!dwfl) return false;

//     Dwarf_Addr addr = static_cast<Dwarf_Addr>(pc);
//     Dwfl_Module* module = dwfl_addrmodule(dwfl, addr);
//     if (!module) { return false; }

//     // Get function name
//     const char* func_name = dwfl_module_addrname(module, addr);
//     if (func_name) {
//       // Demangle C++ name
//       int status;
//       char* demangled = abi::__cxa_demangle(func_name, nullptr, nullptr, &status);
//       if (status == 0 && demangled) {
//         func = demangled;
//         free(demangled);
//       } else {
//         func = func_name;
//       }
//     } else {
//       func = "[unknown function]";
//     }

//     // Get source file and line number
//     Dwfl_Line* dwfl_line = dwfl_module_getsrc(module, addr);
//     if (dwfl_line) {
//       Dwarf_Addr line_addr;
//       int line_num;
//       int col_num;
//       const char* file_name = dwfl_lineinfo(dwfl_line, &line_addr, &line_num, &col_num, nullptr, nullptr);
//       if (file_name) {
//         file = file_name;
//         line = line_num;
//         return true;
//       }
//     }

//     file = "[unknown file]";
//     line = 0;
//     return true;
//   }

// private:
//   Dwfl* dwfl = nullptr;
// };

// // The main function to print the stack trace
// void print_stack_trace_with_dwarf() {
//   DwarfResolver resolver;
//   unw_cursor_t cursor;
//   unw_context_t context;

//   unw_getcontext(&context);
//   unw_init_local(&cursor, &context);

//   std::cout << "--- C++ Call Stack (with DWARF info) ---" << std::endl;
//   int frame_num = 0;

//   while (unw_step(&cursor) > 0) {
//     unw_word_t ip = 0;
//     unw_get_reg(&cursor, UNW_REG_IP, &ip);

//     if (ip == 0) { break; }

//     std::string file, func;
//     int line;

//     // We subtract 1 from the instruction pointer because the IP from a call
//     // instruction points to the instruction *after* the call. The debug info
//     // for the call itself is associated with the call instruction's address.
//     if (resolver.resolve_address(static_cast<uintptr_t>(ip) - 1, file, line, func)) {
//       std::cout << "#" << frame_num << ": " << func << " at " << file << ":" << line << " (0x" << std::hex << ip << std::dec << ")" << std::endl;
//     } else {
//       std::cout << "#" << frame_num << ": [unable to resolve] (0x" << std::hex << ip << std::dec << ")" << std::endl;
//     }
//     frame_num++;
//   }
//   std::cout << "------------------------------------------" << std::endl;
// }

// static void crash_handler(int sig) {
//   std::cout << "\n*** CRASH DETECTED ***" << std::endl;
//   std::cout << "Signal: " << sig << " (" << strsignal(sig) << ")" << std::endl;
//   print_stack_trace_with_dwarf();
//   std::cout << "\n*** END CRASH INFO ***" << std::endl;
//   exit(sig);
// }
// static void setup_crash_handler() {
//   signal(SIGSEGV, crash_handler);
//   signal(SIGABRT, crash_handler);
//   signal(SIGFPE, crash_handler);
//   signal(SIGILL, crash_handler);
//   signal(SIGBUS, crash_handler);
// }

// // 测试代码
// class TestClass {
// private:
//   int* dangerous_ptr;

// public:
//   TestClass() : dangerous_ptr(nullptr) {}

//   void method_level_3() {
//     std::cout << "About to crash in method_level_3..." << std::endl;
//     // print_stack_trace_with_dwarf();
//     // 这里会产生段错误，能看到具体的文件和行号
//     *dangerous_ptr = 42;
//   }

//   void method_level_2() {
//     std::cout << "In method_level_2" << std::endl;
//     method_level_3();
//   }

//   void method_level_1() {
//     std::cout << "In method_level_1" << std::endl;
//     method_level_2();
//   }

//   template <typename T>
//   void template_method(T value) {
//     std::cout << "Template method with value: " << value << std::endl;
//     method_level_1();
//   }
// };

// void global_function_c() {
//   TestClass obj;
//   obj.template_method(123);
// }

// void global_function_b() {
//   std::cout << "In global_function_b" << std::endl;
//   global_function_c();
// }

// void global_function_a() {
//   std::cout << "In global_function_a" << std::endl;
//   global_function_b();
// }

// template <int parameter>
// void test_lambda_crash() {
//   auto lambda = [&](int x) {
//     std::cout << "template parameter " << parameter << " Lambda with parameter: " << x << std::endl;
//     global_function_a();
//   };

//   lambda(456);
// }

// int main() {
//   setup_crash_handler();
//   test_lambda_crash<0>();
//   return 0;
// }

#include <cxxabi.h> // 用于__cxa_demangle
#include <cxxabi.h>
#include <elfutils/libdwfl.h> // 用于libdwfl解析调试信息, sudo apt install libdw
#include <unistd.h> // for getpid()
#include <unwind.h> // 用于_Unwind_Backtrace

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// 栈帧状态结构体，用于_Unwind_Backtrace回调
struct BacktraceState {
  std::vector<uintptr_t> frames;
  size_t max_frames;
};

// _Unwind_Backtrace的回调函数，用于收集栈帧PC地址
static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context* context, void* arg) {
  BacktraceState* state = static_cast<BacktraceState*>(arg);
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    state->frames.push_back(pc);
    if (state->frames.size() >= state->max_frames) { return _URC_END_OF_STACK; }
  }
  return _URC_NO_REASON;
}

// DebugInfoSession：初始化libdwfl会话，用于进程报告
struct DebugInfoSession {
  Dwfl_Callbacks callbacks = {};
  char* debuginfo_path = nullptr;
  Dwfl* dwfl = nullptr;

  DebugInfoSession() {
    callbacks.find_elf = dwfl_linux_proc_find_elf;
    callbacks.find_debuginfo = dwfl_standard_find_debuginfo;
    callbacks.debuginfo_path = &debuginfo_path;

    dwfl = dwfl_begin(&callbacks);
    assert(dwfl != nullptr);

    int r = dwfl_linux_proc_report(dwfl, getpid());
    assert(r == 0);
    r = dwfl_report_end(dwfl, nullptr, nullptr);
    assert(r == 0);
  }

  ~DebugInfoSession() { dwfl_end(dwfl); }

  DebugInfoSession(const DebugInfoSession&) = delete;
  DebugInfoSession& operator=(const DebugInfoSession&) = delete;
};

// DebugInfo：为每个栈帧地址解析函数名、源文件和行号
struct DebugInfo {
  uintptr_t addr;
  std::string function;
  std::string file;
  int line;

  DebugInfo(const DebugInfoSession& dis, uintptr_t addr) : addr(addr), line(-1) {
    Dwfl_Module* module = dwfl_addrmodule(dis.dwfl, addr);
    if (module) {
      // 获取函数名
      const char* name = dwfl_module_addrname(module, addr);
      if (name) {
        int status = -1;
        char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        function = (status == 0) ? demangled : name;
        free(demangled);
      } else {
        function = "<unknown>";
      }

      // 获取源文件和行号
      Dwfl_Line* dwfl_line = dwfl_module_getsrc(module, addr);
      if (dwfl_line) {
        Dwarf_Addr dwarf_addr;
        const char* src_file = dwfl_lineinfo(dwfl_line, &dwarf_addr, &line, nullptr, nullptr, nullptr);
        if (src_file) { file = src_file; }
      }
    } else {
      function = "<unknown>";
      file = "";
    }
  }
};

// 打印栈跟踪函数
void print_stacktrace(size_t max_frames = 32) {
  BacktraceState state;
  state.max_frames = max_frames;

  // 使用_Unwind_Backtrace收集栈帧
  _Unwind_Backtrace(unwind_callback, &state);

  if (state.frames.empty()) {
    std::cerr << "No stack frames found.\n";
    return;
  }

  DebugInfoSession dis;

  std::cerr << "Stacktrace of " << state.frames.size() << " frames:\n";
  for (size_t i = 0; i < state.frames.size(); ++i) {
    DebugInfo di(dis, state.frames[i]);
    std::cerr << "#" << i << ": " << di.addr << " " << di.function;
    if (!di.file.empty()) { std::cerr << " at " << di.file << ":" << di.line; }
    std::cerr << "\n";
  }
  std::cerr.flush();
}

// 示例：异常处理函数
void handle_terminate() {
  print_stacktrace();
  std::_Exit(EXIT_FAILURE);
}

int main() {
  std::set_terminate(handle_terminate);
  // 测试：抛出异常
  throw std::runtime_error("Test exception");
  return 0;
}
