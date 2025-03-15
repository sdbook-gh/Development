#include "lib.h"

NM::NMClass::NMClass() { _NM_value = 0; }
void NM::NMClass::set_NM_value(int value) {
  _NM_value = value;
  increment_NM_value();
}
namespace NM {
void NMClass::increment_NM_value() const { _NM_value++; }
} // namespace NM

std::shared_ptr<open::nmutils::nmlog::NMAsyncLog> NM::NMClass::adjust_NM_log(std::shared_ptr<open::nmutils::nmlog::NMAsyncLog> log, std::shared_ptr<NM::NMAnotherClass> another) {
  NM::NMClass::Enum e1;
  NM::NMClass::EnumClass e2;
  std::vector<int> vec1;
  std::vector<open::NM_managed_ptr<int>> vec2;
  auto ptr = another->get_instance<open::nmutils::nmlog::NMAsyncLog*>(log.get());
  return std::make_shared<open::nmutils::nmlog::NMAsyncLog>();
}
