load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_cc_binary", "apollo_package", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "receiver_files",
    srcs = glob([
        "dag/**",
        "launch/**",
        "conf/**",
    ]),
)

apollo_component(
    name = "libreceiver_component.so",
    srcs = [
        "receiver_component.cc",
    ],
    hdrs = [
        "receiver_component.h",
    ],
    linkstatic = True,
    deps = [
        "//cyber",
        "//component/receiver/proto:receiver_proto",

    ],
)

apollo_package()

cpplint()