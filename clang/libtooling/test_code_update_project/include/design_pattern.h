#pragma once

#include "def.h"
#include <vector>

BEGIN_NS

namespace nm_factory {

class NSDEFObserver {
public:
  virtual void ns_update() = 0;
};

class NSDEFSubject {
public:
  typedef NSDEFObserver OBSERVER;
  typedef std::vector<NSDEFObserver *>::iterator OBSERVER_ITERATOR;

private:
  std::vector<OBSERVER *> ns_observers;

public:
  void ns_attach(OBSERVER *observer);

  void ns_detach(OBSERVER *observer);

  void ns_notify();
};

} // namespace nm_factory

END_NS
