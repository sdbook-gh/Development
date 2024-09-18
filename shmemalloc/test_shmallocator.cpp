#include "shmallocator.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "spdlog/common.h"
#include "spdlog/sinks/basic_file_sink.h"

int main(int argc, const char *const argv[]) {
  auto old_logger = spdlog::default_logger();
  auto new_logger = spdlog::basic_logger_mt("basic_logger", "log.txt", true);
  spdlog::set_default_logger(new_logger);
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::err);
  // spdlog::set_pattern("[%H:%M:%S %z] [%^%L%$] [thread %t] %v");
  spdlog::set_pattern("[source %s] [function %!] [line %#] %v");

  size_t MEM_POOL_SIZE = 100ul * 1024ul * 1024ul; // 100M
  size_t MEM_POOL_BASE_ADDR = 123ul * 1024ul * 1024ul * 1024ul; // 123G
  struct Node {
    int val{0};
    shmallocator::shmstring str{""};
    shmallocator::shmvector<shmallocator::shmstring> str_vec{};
  };
  typedef struct {
    shmallocator::ObjectTag tag{0};
    shmallocator::shmvector<Node> vec;
  } Vector_t;
  typedef struct {
    shmallocator::ObjectTag tag{0};
    shmallocator::shmmutex mutex;
  } Mutex_t;
  typedef struct {
    shmallocator::ObjectTag tag{0};
    shmallocator::shmsemaphore semaphore;
  } Semaphore_t;
  typedef struct {
    shmallocator::ObjectTag tag{0};
    shmallocator::shmcond cond;
  } Cond_t;
  typedef struct {
    shmallocator::ObjectTag tag{0};
    shmallocator::shmqueue<Node> queue{10};
  } Queue_t;
  typedef struct {
    uint32_t id{0};
    shmallocator::ObjectTag tag{0};
    shmallocator::shmmutex mutex;
    shmallocator::slab::SlabManager manager;
  } Slab_t;
  // typedef struct {
  //   shmallocator::ObjectTag tag{0};
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
    shmallocator::start_alive_monitor();
  } else if (argc == 2 && std::string("producer") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    shmallocator::start_heart_beat();
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
    Slab_t *slabptr = shmallocator::shmgetobjbytag<Slab_t>("slab", true);
    if (slabptr == nullptr) {
      printf("slabptr null\n");
      return -1;
    }
    // auto *pslab = &slabptr->manager;
    // printf("pslab: %p\n", pslab);
    // std::vector<std::thread> th_vec;
    // std::set<uint32_t *> set;
    // shmallocator::SlabCache *pcache{nullptr};
    // std::mutex mutex;
    // const auto INSERT_THREAD_COUNT = 5u;
    // const auto ERASE_THREAD_COUNT = 5u;
    // std::atomic_uint8_t insert_thread_completed{0};
    // for (auto i = 0u; i < INSERT_THREAD_COUNT; i++) {
    //   th_vec.emplace_back(std::thread{[&mutex, pslab, &set, &pcache, &insert_thread_completed] {
    //     for (auto i = 0; i < 500; i++) {
    //       uint32_t *pval;
    //       {
    //         shmallocator::SlabCache *cache;
    //         pval = pslab->cache_alloc<uint32_t>(cache);
    //         {
    //           std::lock_guard<std::mutex> lock{mutex};
    //           set.insert(pval);
    //           if (cache != pcache) {
    //             if (pcache == nullptr) {
    //               pcache = cache;
    //             } else {
    //               SPDLOG_ERROR("cache diff {} {}\n", (void *)cache, (void *)pcache);
    //             }
    //           }
    //         }
    //       }
    //     }
    //     insert_thread_completed++;
    //   }});
    // }
    // for (auto i = 0u; i < ERASE_THREAD_COUNT; i++) {
    //   th_vec.emplace_back(std::thread{[&mutex, pslab, &set, &pcache, &insert_thread_completed] {
    //     while (true) {
    //       uint32_t *ptr{nullptr};
    //       {
    //         std::lock_guard<std::mutex> lock{mutex};
    //         if (set.empty() == false) {
    //           ptr = *set.begin();
    //           set.erase(set.begin());
    //         } else if (insert_thread_completed < INSERT_THREAD_COUNT) {
    //           continue;
    //         } else {
    //           break;
    //         }
    //       }
    //       pslab->cache_free(pcache, ptr);
    //     }
    //   }});
    // }
    // std::for_each(th_vec.begin(), th_vec.end(), [&](auto &e) { e.join(); });
    // printf("slab set size %lu\n", set.size());

    // pslab->clear();
    // shmallocator::shmfree(slabptr);
    // SPDLOG_INFO("slab deallocate completed");

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
    // while (true) {
    //   {
    //     std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
    //     shmallocator::shmvector<shmallocator::shmstring> vec{};
    //     vec.push_back("str_vec");
    //     pvec->emplace_back(Node{1, "str", vec});
    //     pcond->signal();
    //   }
    //   printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
    //   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // }
    pqueue->registerProducer("P1");
    while (true) {
      if (!pqueue->push(Node{.val = 1, .str = "producer", .str_vec = std::vector<std::string>{"1"}})) {
        printf("queue is full %u\n", pqueue->size());
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    printf("producer completed\n");
  } else if (argc == 2 && std::string("consumer") == argv[1]) {
    shmallocator::initshm("/tmp/test_shmem", MEM_POOL_BASE_ADDR, MEM_POOL_SIZE);
    shmallocator::start_heart_beat();
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

    // ready go
    // std::vector<std::thread> thread_vec;
    // for (int i = 0; i < 10; ++i) {
    //   thread_vec.emplace_back(std::thread{[&] {
    //     // while (true) {
    //     //   {
    //     //     printf("thread %lu wait\n", pthread_self());
    //     //     psemaphore->wait();
    //     //     printf("thread %lu running\n", pthread_self());
    //     //     {
    //     //       std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
    //     //       if (pvec->size() > 0) {
    //     //         printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(),
    //     (*pvec)[0].str_vec[0].c_str());
    //     //         pvec->erase(pvec->begin());
    //     //         printf("thread %lu got\n", pthread_self());
    //     //       } else {
    //     //         printf("thread %lu empty\n", pthread_self());
    //     //       }
    //     //     }
    //     //   }
    //     // }
    //     auto check = [pvec] {
    //       if (pvec->size() > 0) {
    //         printf("pvec[0]: %d %s %s\n", (*pvec)[0].val, (*pvec)[0].str.c_str(), (*pvec)[0].str_vec[0].c_str());
    //         pvec->erase(pvec->begin());
    //         printf("thread %lu got\n", pthread_self());
    //         return true;
    //       }
    //       return false;
    //     };
    //     // while (true) {
    //     //   bool need_wait{false};
    //     //   {
    //     //     shmallocator::shm_spin_mutex_lock lock{*pspinmutex};
    //     //     if (!check()) {
    //     //       printf("thread %lu wait\n", pthread_self());
    //     //       // pspincond->wait(lock);
    //     //       // printf("thread %lu running\n", pthread_self());
    //     //       // check();
    //     //       need_wait = true;
    //     //     }
    //     //   }
    //     //   if (need_wait) {
    //     //     psemaphore->wait();
    //     //   }
    //     // }
    //     while (true) {
    //       {
    //         std::lock_guard<shmallocator::shmmutex> lock{*pmutex};
    //         if (!check()) {
    //           printf("thread %lu wait\n", pthread_self());
    //           pcond->wait(*pmutex);
    //         }
    //       }
    //     }
    //   }});
    //   // std::cin.ignore();
    //   // thread_vec.rbegin()->detach();
    // }
    // std::for_each(thread_vec.begin(), thread_vec.end(), [](auto &th) { th.join(); });
    pqueue->registerConsumer("C1");
    while (true) {
      Node node;
      if (!pqueue->pop(node)) {
        printf("queue is empty %u\n", pqueue->size());
      } else {
        printf("thread %lu got node %d %s %u\n", pthread_self(), node.val, node.str.c_str(), pqueue->size());
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    printf("consumer completed\n");
  }
  return 0;
}
