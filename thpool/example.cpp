#include <cstdio>
#include <algorithm>
#include <thread>
#include "thpool.h"

void task(void *arg) {
    printf("Thread #%lu working on %d\n", (unsigned long)std::hash<std::thread::id>{}(std::this_thread::get_id()), (int) arg);
}


int main() {
    // std::thread th{[&]{std::this_thread::sleep_for(std::chrono::seconds{1});printf("run in thread\n");}};
    // th.detach();
    printf("Making threadpool with 4 threads\n");
    threadpool thpool = thpool_init(4);

    puts("Adding 40 tasks to threadpool");
    int i;
    for (i=0; i<40; i++){
        thpool_add_work(thpool, task, (void*)i);
    };

    printf("waiting threadpool\n");
    thpool_wait(thpool);
    printf("Killing threadpool\n");
    thpool_destroy(thpool);
    return 0;
}
