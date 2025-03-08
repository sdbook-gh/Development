#include "design_pattern.h"
#include "lib.h"
#include "log.h"
#include "stack_trace.h"
#include <cstdio>

bool check_NM_value(NM::NMClass const &nmc) {
  if (nmc.get_NM_value() == 0) {
    return true;
  }
  return false;
}

class MyObserver : public NM::nm_factory::NSDEFObserver {
public:
  void ns_update() override { printf("MyObserver::ns_update\n"); }
};
class MySubject : public NM::nm_factory::NSDEFSubject {
public:
  void ns_notify() { printf("MySubject::ns_notify\n"); }
};

class MyStackTracker : public open::nmutils::nmtrace::NMStackTrace {
public:
  void GetStack() {
    auto stackTrace = GetNMStackTrace();
    for (auto &frame : stackTrace) {
      printf("Frame: %s\n", frame.c_str());
    }
  }
};

int main() {
  using namespace NM;
  using namespace open::nmutils::nmlog;
  NMClass nmc;
  nmc.set_NM_value(0);
  nmc.get_NM_value();
  open::nmutils::nmlog::NMAsyncLog *ptr = new open::nmutils::nmlog::NMAsyncLog();
  NMAsyncLog log;
  log.addErrorHandler([&nmc](const std::string &msg) {
    printf("Error: %s\n", msg.c_str());
    nmc.set_NM_value(-1);
    open::nmutils::nmtrace::NMStackTrace nmst;
    auto stackTrace = nmst.GetNMStackTrace();
  });
  nm_values::NS_PREFIX(TestEnum1) e1 = nm_values::NS_PREFIX(ENUM_1);
  nm_values::NMTestEnum2 e2 = nm_values::NMTestEnum2::NM_ENUM_4;
  return 0;
}
