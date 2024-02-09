#include <vector>
#include <cstdio>

int run_lib();

int main(void) {
  auto res = run_lib();
  std::vector<int> vec;
  printf("%d\n", vec[0]);
  return 0;
}
