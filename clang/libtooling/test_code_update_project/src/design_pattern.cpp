#include "design_pattern.h"
#include <algorithm>

BEGIN_NS

void nm_factory::NSDEFSubject::ns_attach(OBSERVER *observer) {
  this->ns_observers.push_back(observer);
}

void nm_factory::NSDEFSubject::ns_detach(OBSERVER *observer) {
  OBSERVER_ITERATOR it =
      std::find(this->ns_observers.begin(), this->ns_observers.end(), observer);
  if (it != this->ns_observers.end()) {
    this->ns_observers.erase(it);
  }
}

void nm_factory::NSDEFSubject::ns_notify() {
  for (auto observer : ns_observers) {
    observer->ns_update();
  }
}

END_NS
