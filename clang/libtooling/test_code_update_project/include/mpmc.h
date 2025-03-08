#pragma once

#include <atomic>

struct NMNode {
  void *data;
  std::atomic<NMNode *> next;

  NMNode() : data(nullptr), next(nullptr) {}
  explicit NMNode(void *d) : data(d), next(nullptr) {}
};

class NMLockFreeMPMCQueue {
public:
  NMLockFreeMPMCQueue();
  ~NMLockFreeMPMCQueue();

  void enqueue(void *item);
  bool dequeue(void *&result);

private:
  std::atomic<NMNode *> m_head;
  std::atomic<NMNode *> m_tail;
};
