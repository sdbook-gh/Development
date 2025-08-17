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
