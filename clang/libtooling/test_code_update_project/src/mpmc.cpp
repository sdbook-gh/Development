#include "mpmc.h"

NMLockFreeMPMCQueue::NMLockFreeMPMCQueue() {
  NMNode *dummy = new NMNode();
  m_head.store(dummy, std::memory_order_relaxed);
  m_tail.store(dummy, std::memory_order_relaxed);
}

NMLockFreeMPMCQueue::~NMLockFreeMPMCQueue() {
  while (NMNode *old_head = m_head.load()) {
    m_head.store(old_head->next);
    delete old_head;
  }
}

void NMLockFreeMPMCQueue::enqueue(void *item) {
  NMNode *newNMNode = new NMNode(item);
  NMNode *oldTail = m_tail.load(std::memory_order_relaxed);

  while (true) {
    NMNode *next = oldTail->next.load(std::memory_order_acquire);
    if (next == nullptr) {
      if (oldTail->next.compare_exchange_weak(next, newNMNode,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
        m_tail.compare_exchange_weak(oldTail, newNMNode,
                                     std::memory_order_release,
                                     std::memory_order_relaxed);
        return;
      }
    } else {
      m_tail.compare_exchange_weak(oldTail, next, std::memory_order_release,
                                   std::memory_order_relaxed);
    }
  }
}

bool NMLockFreeMPMCQueue::dequeue(void *&result) {
  NMNode *oldHead = m_head.load(std::memory_order_relaxed);

  while (true) {
    NMNode *next = oldHead->next.load(std::memory_order_acquire);
    if (next == nullptr)
      return false;

    if (m_head.compare_exchange_weak(oldHead, next, std::memory_order_release,
                                     std::memory_order_relaxed)) {
      result = next->data;
      delete oldHead;
      return true;
    }
  }
}
