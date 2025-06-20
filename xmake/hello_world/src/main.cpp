#include <iostream>
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"

int main(int argc, char **argv) {
  // Set default logger to stdout with info level
  spdlog::set_level(spdlog::level::info);
  auto console = spdlog::stdout_color_mt("console");
  spdlog::set_default_logger(console);
  spdlog::info("hello world!");
  return 0;
}
