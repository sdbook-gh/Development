#include <cstdio>

#define xstr(s) str(s)
#define str(s) #s
#define pr_fmt(fmt) "test_macro: " __FILE__ " " xstr(__LINE__) ": " fmt
#define pr_debug(fmt, ...) fprintf(stdout, pr_fmt(fmt), ##__VA_ARGS__)

int main() {
  pr_debug("Hello world! %s\n", "test");
  return 0;
}
