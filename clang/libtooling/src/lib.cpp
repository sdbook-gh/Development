#include "lib.h"

NM::NMClass::NMClass() { _NM_value = 0; }
void NM::NMClass::set_NM_value(int value) { _NM_value = value; }
namespace NM {
void NMClass::increment_NM_value() { _NM_value++; }
} // namespace NM
