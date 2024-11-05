#include "gtest/gtest.h"
#include "test_so/test_so.h"
#include <cstdio>

TEST(test,content)
{
  printf("get_val %d\n", get_val());
  EXPECT_EQ("1","Hello World!");
}

int main(int argc, char** argv) {
  return RUN_ALL_TESTS();
}
