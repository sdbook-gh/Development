#include "design_pattern.h"
#include "lib.h"
#include "log.h"
#include "stack_trace.h"
#include <cstdio>
#include <vector>
#include <map>

bool check_NM_value(NM::NMClass const &nmc) {
  if (nmc.get_NM_value() == 0) {
    return true;
  }
  return false;
}

open::nmutils::nmtrace::NMStackTrace get_stack_trace(const open::nmutils::nmtrace::NMStackTrace &st,NM::nm_values::NMTestEnum2 e2) {
  return open::nmutils::nmtrace::NMStackTrace{};
}

std::vector<open::nmutils::nmtrace::NMStackTrace> stack_trace_vec;

class MyObserver : public NM::nm_factory::NSDEFObserver {
public:
  void ns_update() override { printf("MyObserver::ns_update\n"); }
};
class MySubject : public NM::nm_factory::NSDEFSubject {
public:
  void ns_notify() { printf("MySubject::ns_notify\n"); }
};
std::map<NM::nm_values::NS_PREFIX(OpenEnum), std::vector<NM::NMClass*>*> nm_map;

class MyStackTracker : public open::nmutils::nmtrace::NMStackTrace {
public:
  void GetStack() {
    auto stackTrace = GetNMStackTrace();
    for (auto &frame : stackTrace) {
      printf("Frame: %s\n", frame.c_str());
    }
  }
};

open::managed_ptr<open::nmutils::nmtrace::NMStackTrace> ptr1;
open::managed_ptr<NM::NMClass> ptr2;

int main() {
  using namespace NM;
  using namespace open::nmutils::nmlog;
  NMClass nmc;
  nmc.set_NM_value(0);
  nmc.get_NM_value();
  open::nmutils::nmlog::NMAsyncLog *ptr =
      new open::nmutils::nmlog::NMAsyncLog();
  NMAsyncLog log;
  log.addErrorHandler([&nmc](const std::string &msg) {
    printf("Error: %s\n", msg.c_str());
    nmc.set_NM_value(-1);
    open::nmutils::nmtrace::NMStackTrace nmst;
    auto stackTrace = nmst.GetNMStackTrace();
  });
  nm_values::NS_PREFIX(TestEnum1) e1 = nm_values::NS_PREFIX(ENUM_1);
  nm_values::NMTestEnum2 e2 = nm_values::NMTestEnum2::NM_ENUM_4;
  using namespace nm_values;
  NS_PREFIX(OpenEnum) e3 = NS_PREFIX(OpenEnum)::ALL;
  using MT = nm_values::NMTestEnum2;
  return 0;
}
