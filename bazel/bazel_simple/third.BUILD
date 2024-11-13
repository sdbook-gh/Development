cc_library(
    name = "test_so",
    srcs = glob(["test_so/*so*"]),
    hdrs = glob(["test_so/*h*"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest",
    hdrs = glob(["**/*.h",]),
    srcs = glob(["gtest/lib/libgtest*so*"]),
    includes = ["gtest/include",],
    visibility = ["//visibility:public"],
)
