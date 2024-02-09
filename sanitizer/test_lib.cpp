#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

static uint16_t g_flag{0}; // atomic_uint16_t

int run_lib() {
  int a[8] = {0};
  int b[8] = {0};
  b[1] = 1;
  for (int i = 0, x = 0, y = 0; i <= 8; ++i) { // < 8
    b[i] += x + y;
    y = x;
    x = b[i];
  }
  new uint8_t[16];
  std::vector<std::thread> vec_thread;
  for (int i = 0; i < 16; ++i) {
    vec_thread.emplace_back(std::thread{[] { g_flag += 2; }});
  }
  while (true) {
    if (g_flag >= 16) {
      break;
    }
  }
  std::for_each(vec_thread.begin(), vec_thread.end(), [](auto &th) { th.join(); });
  return a[0];
}
