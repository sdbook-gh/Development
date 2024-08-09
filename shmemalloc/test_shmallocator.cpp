#include "shmallocator.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

struct Node {
  int val{0};
  shmallocator::shmstring str{""};
  shmallocator::shmvector<shmallocator::shmstring> str_vec{};
};

int main(int argc, const char *const argv[]) {
  size_t MEM_POOL_SIZE = 10ul * 1024ul * 1024ul; // 10M
  size_t MEM_POOL_BASE_ADDR = 123ul * 1024ul * 1024ul * 1024ul; // 123G
  typedef struct {
    char tag[64]{0};
    shmallocator::shmvector<Node> vec;
  } Vector_t;
  typedef struct {
    char tag[64]{0};
    shmallocator::shmmutex mutex;
  } Mutex_t;
  typedef struct {
    char tag[64]{0};
    shmallocator::shmcond mcond;
  } Cond_t;
  if (argc == 2 && std::string("clearer") == argv[1]) {
    if (std::filesystem::remove("/tmp/test_shmem")) {
      printf("/tmp/test_shmem is cleared\n");
    } else {
      printf("clear /tmp/test_shmem error\n");
    }
  } else if (argc == 2 && std::string("producer") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    auto *vecptr = shmallocator::shmgetobjbytag<Vector_t>("vec");
    auto *pvec = &vecptr->vec;
    printf("pvec: %p\n", pvec);
    Mutex_t *mutexptr = shmallocator::shmgetobjbytag<Mutex_t>("mutex");
    auto *pmutex = &mutexptr->mutex;
    printf("pmutex: %p\n", pmutex);
    Cond_t *condptr = shmallocator::shmgetobjbytag<Cond_t>("cond");
    auto *pcond = &condptr->mcond;
    printf("pcond: %p\n", pcond);
    // auto *pmutex = (shmallocator::shmmutex *)((char *)shmallocator::shmptr + 11280);
    // new (pmutex) shmallocator::shmmutex;
    // auto *pcond = (shmallocator::shmcond *)((char *)shmallocator::shmptr + 12560);
    // new (pcond) shmallocator::shmcond;
    while (true) {
      {
        std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
        // pmutex->lock();
        shmallocator::shmvector<shmallocator::shmstring> vec{};
        vec.push_back("str_vec");
        pvec->emplace_back(Node{1, "str", vec});
        pcond->signal();
        // pmutex->unlock();
      }
      printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    printf("producer completed\n");
  } else if (std::string("consumer") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    auto *vecptr = shmallocator::shmgetobjbytag<Vector_t>("vec");
    auto *pvec = &vecptr->vec;
    printf("pvec: %p\n", pvec);
    Mutex_t *mutexptr = shmallocator::shmgetobjbytag<Mutex_t>("mutex");
    auto *pmutex = &mutexptr->mutex;
    printf("pmutex: %p\n", pmutex);
    Cond_t *condptr = shmallocator::shmgetobjbytag<Cond_t>("cond");
    auto *pcond = &condptr->mcond;
    printf("pcond: %p\n", pcond);
    // auto *pmutex = (shmallocator::shmmutex *)((char *)shmallocator::shmptr + 11280);
    // auto *pcond = (shmallocator::shmcond *)((char *)shmallocator::shmptr + 12560);
    while (true) {
      {
        std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
        // pmutex->lock();
        pcond->wait(*pmutex);
        if (pvec->size() > 0) {
          printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
          pvec->erase(pvec->begin());
        } else {
          printf("pvec empty\n");
        }
        // pmutex->unlock();
      }
      // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    printf("consumer completed\n");
  }
  return 0;
}
