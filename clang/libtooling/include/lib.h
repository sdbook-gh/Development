#pragma once

namespace NM {
class NMClass {
public:
  NMClass();
  void set_NM_value(int value);
  int get_NM_value() {
    increment_NM_value();
    return _NM_value;
  }

private:
  void increment_NM_value();
  int _NM_value;
};
} // namespace NM
