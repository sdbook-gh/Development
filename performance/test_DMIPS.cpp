#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <cstring>
#ifdef __linux__
#include <unistd.h>
#ifdef ANDROID
#include <sched.h>
#include <sys/syscall.h>
#else
#include <pthread.h>
#endif
#endif
#ifdef _WIN32
#include <windows.h>
#endif

// Dhrystone基准测试相关定义
#define LOOPS 50000000 // 默认循环次数

// 记录类型定义
enum Enumeration { Ident_1, Ident_2, Ident_3, Ident_4, Ident_5 };

typedef struct record {
  struct record *Ptr_Comp;
  Enumeration Discr;
  union {
    struct {
      Enumeration Enum_Comp;
      int Int_Comp;
      char Str_Comp[31];
    } var_1;
    struct {
      Enumeration E_Comp_2;
      char Str_2_Comp[31];
    } var_2;
    struct {
      char Ch_1_Comp;
      char Ch_2_Comp;
    } var_3;
  } variant;
} Rec_Type, *Rec_Pointer;

// 全局变量
Rec_Pointer Ptr_Glob, Next_Ptr_Glob;
int Int_Glob;
bool Bool_Glob;
char Ch_1_Glob, Ch_2_Glob;
int Arr_1_Glob[50];
int Arr_2_Glob[50][50];

// 函数声明
void Proc_1(Rec_Pointer Ptr_Val_Par);
void Proc_2(int *Int_Par_Ref);
void Proc_3(Rec_Pointer *Ptr_Ref_Par);
void Proc_4();
void Proc_5();
void Proc_6(Enumeration Enum_Val_Par, Enumeration *Enum_Ref_Par);
void Proc_7(int Int_1_Par_Val, int Int_2_Par_Val, int *Int_Par_Ref);
void Proc_8(int Arr_1_Par_Ref[50], int Arr_2_Par_Ref[50][50], int Int_1_Par_Val, int Int_2_Par_Val);
Enumeration Func_1(char Ch_1_Par_Val, char Ch_2_Par_Val);
bool Func_2(char Str_1_Par_Ref[31], char Str_2_Par_Ref[31]);
bool Func_3(Enumeration Enum_Par_Val);

class DMIPSTester {
public:
  DMIPSTester() : loops(LOOPS) {}

  void setLoops(int numLoops) { loops = numLoops; }

  double runBenchmark() {
    initializeGlobals();

    auto start = std::chrono::high_resolution_clock::now();

    // 主测试循环
    for (int i = 0; i < loops; ++i) {
      Proc_5();
      Proc_4();

      int Int_1_Loc = 2;
      int Int_2_Loc = 3;
      char Str_1_Loc[31] = "DHRYSTONE PROGRAM, 1'ST STRING";
      char Str_2_Loc[31] = "DHRYSTONE PROGRAM, 2'ND STRING";
      Enumeration Enum_Loc = Ident_2;

      Bool_Glob = !Func_2(Str_1_Loc, Str_2_Loc);
      int Int_3_Loc = 0;
      while (Int_1_Loc < Int_2_Loc) {
        Int_3_Loc = 5 * Int_1_Loc - Int_2_Loc;
        Proc_7(Int_1_Loc, Int_2_Loc, &Int_3_Loc);
        Int_1_Loc += 1;
      }

      Proc_8(Arr_1_Glob, Arr_2_Glob, Int_1_Loc, Int_3_Loc);
      Proc_1(Ptr_Glob);

      for (char Ch_Index = 'A'; Ch_Index <= Ch_2_Glob; ++Ch_Index) {
        if (Enum_Loc == Func_1(Ch_Index, 'C')) {
          Proc_6(Ident_1, &Enum_Loc);
          strcpy(Str_2_Loc, "DHRYSTONE PROGRAM, 3'RD STRING");
          Int_2_Loc = i;
          Int_Glob = i;
        }
      }

      Int_2_Loc = Int_2_Loc * Int_1_Loc;
      Int_1_Loc = Int_2_Loc / Int_3_Loc;
      Int_2_Loc = 7 * (Int_2_Loc - Int_3_Loc) - Int_1_Loc;
      Proc_2(&Int_1_Loc);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double seconds = duration.count() / 1000000.0;
    double dhrystones_per_second = loops / seconds;
    double dmips = dhrystones_per_second / 1757.0; // VAX 11/780的参考值

    return dmips;
  }

private:
  int loops;

  void initializeGlobals() {
    Ptr_Glob = new Rec_Type;
    Next_Ptr_Glob = new Rec_Type;

    Ptr_Glob->Ptr_Comp = Next_Ptr_Glob;
    Ptr_Glob->Discr = Ident_1;
    Ptr_Glob->variant.var_1.Enum_Comp = Ident_3;
    Ptr_Glob->variant.var_1.Int_Comp = 40;
    strcpy(Ptr_Glob->variant.var_1.Str_Comp, "DHRYSTONE PROGRAM, SOME STRING");
    strcpy(Next_Ptr_Glob->variant.var_1.Str_Comp, "DHRYSTONE PROGRAM, SOME STRING");
    Next_Ptr_Glob->Ptr_Comp = Ptr_Glob;
    Next_Ptr_Glob->Discr = Ident_2;
    Next_Ptr_Glob->variant.var_2.E_Comp_2 = Ident_2;
    strcpy(Next_Ptr_Glob->variant.var_2.Str_2_Comp, "DHRYSTONE PROGRAM, 2'ND STRING");

    Int_Glob = 5;
    Bool_Glob = false;
    Ch_1_Glob = 'A';
    Ch_2_Glob = 'B';

    for (int i = 0; i < 50; ++i) {
      Arr_1_Glob[i] = 0;
      for (int j = 0; j < 50; ++j) { Arr_2_Glob[i][j] = 0; }
    }
  }
};

// Dhrystone程序函数实现
void Proc_1(Rec_Pointer Ptr_Val_Par) {
  Rec_Pointer Next_Record = Ptr_Val_Par->Ptr_Comp;
  *Ptr_Val_Par->Ptr_Comp = *Ptr_Glob;
  Ptr_Val_Par->variant.var_1.Int_Comp = 5;
  Next_Record->variant.var_1.Int_Comp = Ptr_Val_Par->variant.var_1.Int_Comp;
  Next_Record->Ptr_Comp = Ptr_Val_Par->Ptr_Comp;
  Proc_3(&Next_Record->Ptr_Comp);
  if (Next_Record->Discr == Ident_1) {
    Next_Record->variant.var_1.Int_Comp = 6;
    Proc_6(Ptr_Val_Par->variant.var_1.Enum_Comp, &Next_Record->variant.var_1.Enum_Comp);
    Next_Record->Ptr_Comp = Ptr_Glob->Ptr_Comp;
    Proc_7(Next_Record->variant.var_1.Int_Comp, 10, &Next_Record->variant.var_1.Int_Comp);
  } else {
    *Ptr_Val_Par = *Ptr_Val_Par->Ptr_Comp;
  }
}

void Proc_2(int *Int_Par_Ref) {
  int Int_Loc = *Int_Par_Ref + 10;
  int Enum_Loc;
  do {
    if (Ch_1_Glob == 'A') {
      Int_Loc -= 1;
      *Int_Par_Ref = Int_Loc - Int_Glob;
      Enum_Loc = Ident_1;
    }
  } while (Enum_Loc != Ident_1);
}

void Proc_3(Rec_Pointer *Ptr_Ref_Par) {
  if (Ptr_Glob != nullptr) { *Ptr_Ref_Par = Ptr_Glob->Ptr_Comp; }
  Proc_7(10, Int_Glob, &Ptr_Glob->variant.var_1.Int_Comp);
}

void Proc_4() {
  bool Bool_Loc = Ch_1_Glob == 'A';
  Bool_Loc |= Bool_Glob;
  Ch_2_Glob = 'B';
}

void Proc_5() {
  Ch_1_Glob = 'A';
  Bool_Glob = false;
}

void Proc_6(Enumeration Enum_Val_Par, Enumeration *Enum_Ref_Par) {
  *Enum_Ref_Par = Enum_Val_Par;
  if (!Func_3(Enum_Val_Par)) { *Enum_Ref_Par = Ident_4; }
  switch (Enum_Val_Par) {
    case Ident_1:
      *Enum_Ref_Par = Ident_1;
      break;
    case Ident_2:
      if (Int_Glob > 100) {
        *Enum_Ref_Par = Ident_1;
      } else {
        *Enum_Ref_Par = Ident_4;
      }
      break;
    case Ident_3:
      *Enum_Ref_Par = Ident_2;
      break;
    case Ident_4:
    case Ident_5:
      *Enum_Ref_Par = Ident_3;
      break;
  }
}

void Proc_7(int Int_1_Par_Val, int Int_2_Par_Val, int *Int_Par_Ref) {
  int Int_Loc = Int_1_Par_Val + 2;
  *Int_Par_Ref = Int_2_Par_Val + Int_Loc;
}

void Proc_8(int Arr_1_Par_Ref[50], int Arr_2_Par_Ref[50][50], int Int_1_Par_Val, int Int_2_Par_Val) {
  int Int_Loc = Int_1_Par_Val + 5;
  Arr_1_Par_Ref[Int_Loc] = Int_2_Par_Val;
  Arr_1_Par_Ref[Int_Loc + 1] = Arr_1_Par_Ref[Int_Loc];
  Arr_1_Par_Ref[Int_Loc + 30] = Int_Loc;
  for (int Int_Index = Int_Loc; Int_Index <= Int_Loc + 1; ++Int_Index) { Arr_2_Par_Ref[Int_Loc][Int_Index] = Int_Loc; }
  Arr_2_Par_Ref[Int_Loc][Int_Loc - 1] += 1;
  Arr_2_Par_Ref[Int_Loc + 20][Int_Loc] = Arr_1_Par_Ref[Int_Loc];
  Int_Glob = 5;
}

Enumeration Func_1(char Ch_1_Par_Val, char Ch_2_Par_Val) {
  char Ch_1_Loc = Ch_1_Par_Val;
  char Ch_2_Loc = Ch_1_Loc;
  if (Ch_2_Loc != Ch_2_Par_Val) {
    return Ident_1;
  } else {
    Ch_1_Glob = Ch_1_Loc;
    return Ident_2;
  }
}

bool Func_2(char Str_1_Par_Ref[31], char Str_2_Par_Ref[31]) {
  int Int_Loc = 2;
  char Ch_Loc = 'A';
  while (Int_Loc <= 2) {
    if (Func_1(Str_1_Par_Ref[Int_Loc], Str_2_Par_Ref[Int_Loc + 1]) == Ident_1) {
      Ch_Loc = 'A';
      Int_Loc += 1;
    }
  }
  if (Ch_Loc >= 'W' && Ch_Loc < 'Z') { Int_Loc = 7; }
  if (Ch_Loc == 'R') {
    return true;
  } else {
    if (strcmp(Str_1_Par_Ref, Str_2_Par_Ref) > 0) {
      Int_Loc += 7;
      Int_Glob = Int_Loc;
      return true;
    } else {
      return false;
    }
  }
}

bool Func_3(Enumeration Enum_Par_Val) {
  Enumeration Enum_Loc = Enum_Par_Val;
  if (Enum_Loc == Ident_3) {
    return true;
  } else {
    return false;
  }
}

int main() {
  std::cout << "=== CPU单核DMIPS性能测试 ===" << std::endl;
  std::cout << "基于Dhrystone 2.1基准测试" << std::endl;
  std::cout << std::endl;

  DMIPSTester tester;

// 设置CPU亲和性到单核（Linux系统）
#ifdef __linux__
#ifdef ANDROID
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(3, &mask);
  pid_t tid = gettid(); // 获取线程ID
  if (syscall(__NR_sched_setaffinity, tid, sizeof(mask), &mask) == -1) {
    std::cout << "绑定测试到CPU核心3失败" << std::endl;
  } else {
    std::cout << "已绑定测试到CPU核心3" << std::endl;
  }
#else
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(3, &cpuset); // 绑定到CPU核心3
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  std::cout << "已绑定测试到CPU核心3" << std::endl << std::endl;
#endif
#endif
#ifdef _WIN32
  DWORD_PTR affinityMask = 0x04;
  if (SetProcessAffinityMask(GetCurrentProcess(), affinityMask)) {
    printf("成功设置进程亲和性为 CPU 3\n");
  } else {
    printf("设置失败，错误码:%lu\n", GetLastError());
  }
  // 获取当前进程亲和性
  DWORD_PTR processMask, systemMask;
  if (GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask)) {
    printf("当前进程亲和掩码:0x%llx\n", processMask);
    printf("系统可用掩码:0x%llx\n", systemMask);
  }
#endif

  // 不同循环次数的测试
  int test_loops[] = {1000000, 5000000, 10000000, 50000000};
  int num_tests = sizeof(test_loops) / sizeof(test_loops[0]);

  for (int i = 0; i < num_tests; ++i) {
    std::cout << "正在运行测试 " << (i + 1) << "/" << num_tests << " (循环次数: " << test_loops[i] << ")..." << std::endl;

    tester.setLoops(test_loops[i]);
    double dmips = tester.runBenchmark();

    std::cout << "循环次数: " << test_loops[i] << std::endl;
    std::cout << "DMIPS: " << dmips << std::endl;
    std::cout << std::endl;
  }

  // 最终完整测试
  std::cout << "=== 最终完整测试 ===" << std::endl;
  tester.setLoops(100000000); // 1亿次循环
  std::cout << "正在运行最终测试（1亿次循环）..." << std::endl;

  double final_dmips = tester.runBenchmark();

  std::cout << "最终结果:" << std::endl;
  std::cout << "DMIPS: " << final_dmips << std::endl;
  std::cout << "测试完成" << std::endl;

  return 0;
}
