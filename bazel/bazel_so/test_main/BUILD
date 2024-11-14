cc_binary(
    name = "test_main",
    srcs = glob(["*.cpp"]) + glob(["*.h*"]),
    copts = ["-g"],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
        "@third//:test_so",
    ],
)
