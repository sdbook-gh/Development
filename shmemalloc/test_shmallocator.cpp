#include "shmallocator.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

int main(int argc, const char *const argv[]) {
  size_t MEM_POOL_SIZE = 10ul * 1024ul * 1024ul; // 10M
  size_t MEM_POOL_BASE_ADDR = 123ul * 1024ul * 1024ul * 1024ul; // 123G
  struct Node {
    int val{0};
    shmallocator::shmstring str{""};
    shmallocator::shmvector<shmallocator::shmstring> str_vec{};
  };
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
    shmallocator::shmsemaphore semaphore;
  } Semaphore_t;
  typedef struct {
    char tag[64]{0};
    shmallocator::shmcond cond;
  } Cond_t;
  typedef struct {
    char tag[64]{0};
    shmallocator::AliveMonitor monitor;
  } Heartbeat_t;
  typedef struct {
    char tag[64]{0};
    shmallocator::shmqueue<Node> queue{10};
  } Queue_t;
  // typedef struct {
  //   char tag[64]{0};
  //   shmallocator::shm_spin_mutex spin_mutex;
  //   shmallocator::shm_spin_cond spin_cond;
  // } SpinSync_t;

  if (argc == 2 && std::string("clearer") == argv[1]) {
    if (std::filesystem::remove("/tmp/test_shmem")) {
      printf("/tmp/test_shmem is cleared\n");
    } else {
      printf("clear /tmp/test_shmem error\n");
    }
  } else if (argc == 2 && std::string("monitor") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    auto *monitorptr = shmallocator::shmgetobjbytag<Heartbeat_t>("monitor");
    printf("monitorptr: %p\n", monitorptr);
    auto *pmonitor = &monitorptr->monitor;
    printf("pmonitor: %p\n", pmonitor);
    pmonitor->start_monitor();
  } else if (argc == 2 && std::string("producer") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    auto *monitorptr = shmallocator::shmgetobjbytag<Heartbeat_t>("monitor");
    printf("monitorptr: %p\n", monitorptr);
    auto *pmonitor = &monitorptr->monitor;
    printf("pmonitor: %p\n", pmonitor);
    auto *vecptr = shmallocator::shmgetobjbytag<Vector_t>("vec");
    auto *pvec = &vecptr->vec;
    printf("pvec: %p\n", pvec);
    Mutex_t *mutexptr = shmallocator::shmgetobjbytag<Mutex_t>("mutex");
    auto *pmutex = &mutexptr->mutex;
    printf("pmutex: %p\n", pmutex);
    Semaphore_t *semaphoreptr = shmallocator::shmgetobjbytag<Semaphore_t>("semaphore");
    auto *psemaphore = &semaphoreptr->semaphore;
    printf("psemaphore: %p\n", psemaphore);
    Cond_t *condptr = shmallocator::shmgetobjbytag<Cond_t>("cond");
    auto *pcond = &condptr->cond;
    printf("pcond: %p\n", pcond);
    pmonitor->start_heartbeat(shmallocator::AliveMonitor::PRODUCER, "producer_" + std::to_string(time(nullptr)));
    Queue_t *queueptr = shmallocator::shmgetobjbytag<Queue_t>("queue");
    auto *pqueue = &queueptr->queue;
    printf("pqueue: %p\n", pqueue);
    // auto *spinsyncptr = shmallocator::shmgetobjbytag<SpinSync_t>("spinsync");
    // printf("spinsyncptr: %p\n", spinsyncptr);
    // auto *pspinmutex = &spinsyncptr->spin_mutex;
    // auto *pspincond = &spinsyncptr->spin_cond;

    // ready go
    // while (true) {
    //   {
    //     std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
    //     shmallocator::shmvector<shmallocator::shmstring> vec{};
    //     vec.push_back("str_vec");
    //     pvec->emplace_back(Node{1, "str", vec});
    //     psemaphore->signal();
    //   }
    //   printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
    //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // }
    // while (true) {
    //   if (!pqueue->push(Node{.val = 1, .str = "producer", .str_vec = std::vector<std::string>{"1"}})) {
    //     printf("queue is full %u\n", pqueue->size());
    //   }
    //   std::this_thread::sleep_for(std::chrono::seconds(1));
    // }
    // while (true) {
    //   {
    //     shmallocator::shm_spin_mutex_lock lock{*pspinmutex};
    //     shmallocator::shmvector<shmallocator::shmstring> vec{};
    //     vec.push_back("str_vec");
    //     pvec->emplace_back(Node{1, "str", vec});
    //     // pspincond->notify_one();
    //     psemaphore->signal();
    //   }
    //   printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
    //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // }
    while (true) {
      {
        std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
        shmallocator::shmvector<shmallocator::shmstring> vec{};
        vec.push_back("str_vec");
        pvec->emplace_back(Node{1, "str", vec});
        pcond->signal();
      }
      printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    printf("producer completed\n");
  } else if (argc == 2 && std::string("consumer") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    auto *monitorptr = shmallocator::shmgetobjbytag<Heartbeat_t>("monitor");
    printf("monitorptr: %p\n", monitorptr);
    auto *pmonitor = &monitorptr->monitor;
    printf("pmonitor: %p\n", pmonitor);
    auto *vecptr = shmallocator::shmgetobjbytag<Vector_t>("vec");
    auto *pvec = &vecptr->vec;
    printf("pvec: %p\n", pvec);
    Mutex_t *mutexptr = shmallocator::shmgetobjbytag<Mutex_t>("mutex");
    auto *pmutex = &mutexptr->mutex;
    printf("pmutex: %p\n", pmutex);
    Semaphore_t *semaphoreptr = shmallocator::shmgetobjbytag<Semaphore_t>("semaphore");
    auto *psemaphore = &semaphoreptr->semaphore;
    printf("psemaphore: %p\n", psemaphore);
    Cond_t *condptr = shmallocator::shmgetobjbytag<Cond_t>("cond");
    auto *pcond = &condptr->cond;
    printf("pcond: %p\n", pcond);
    Queue_t *queueptr = shmallocator::shmgetobjbytag<Queue_t>("queue");
    auto *pqueue = &queueptr->queue;
    printf("pqueue: %p\n", pqueue);
    // auto *spinsyncptr = shmallocator::shmgetobjbytag<SpinSync_t>("spinsync");
    // printf("spinsyncptr: %p\n", spinsyncptr);
    // auto *pspinmutex = &spinsyncptr->spin_mutex;
    // auto *pspincond = &spinsyncptr->spin_cond;

    pmonitor->start_heartbeat(shmallocator::AliveMonitor::CONSUMER, "consumer_" + std::to_string(time(nullptr)));
    pmonitor->wait_for_any_producer_alive();
    printf("producer alive\n");
    // ready go
    std::vector<std::thread> thread_vec;
    for (int i = 0; i < 10; ++i) {
      thread_vec.emplace_back(std::thread{[&] {
        // while (true) {
        //   {
        //     printf("thread %lu wait\n", pthread_self());
        //     psemaphore->wait();
        //     printf("thread %lu running\n", pthread_self());
        //     {
        //       std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
        //       if (pvec->size() > 0) {
        //         printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
        //         pvec->erase(pvec->begin());
        //         printf("thread %lu got\n", pthread_self());
        //       } else {
        //         printf("thread %lu empty\n", pthread_self());
        //       }
        //     }
        //   }
        // }
        // while (true) {
        //   Node node;
        //   pqueue->pop(node);
        //   printf("thread %lu got node %d %s %u\n", pthread_self(), node.val, node.str.c_str(), pqueue->size());
        // }
        auto check = [pvec] {
          if (pvec->size() > 0) {
            printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
            pvec->erase(pvec->begin());
            printf("thread %lu got\n", pthread_self());
            return true;
          }
          return false;
        };
        // while (true) {
        //   bool need_wait{false};
        //   {
        //     shmallocator::shm_spin_mutex_lock lock{*pspinmutex};
        //     if (!check()) {
        //       printf("thread %lu wait\n", pthread_self());
        //       // pspincond->wait(lock);
        //       // printf("thread %lu running\n", pthread_self());
        //       // check();
        //       need_wait = true;
        //     }
        //   }
        //   if (need_wait) {
        //     psemaphore->wait();
        //   }
        // }
        while (true) {
          {
            std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
            if (!check()) {
              printf("thread %lu wait\n", pthread_self());
              pcond->wait(*pmutex);
            }
          }
        }
      }});
      // thread_vec.rbegin()->detach();
    }
    std::for_each(thread_vec.begin(), thread_vec.end(), [](auto &th) { th.join(); });
    printf("consumer completed\n");
  }
  return 0;
}
