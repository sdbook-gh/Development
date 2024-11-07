load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_cc_binary", "apollo_package", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "sender_files",
    srcs = glob([
        "dag/**",
        "launch/**",
        "conf/**",
    ]),
)

apollo_component(
    name = "libsender_component.so",
    srcs = [
        "sender_component.cc",
    ],
    hdrs = [
        "sender_component.h",
    ],
    linkstatic = True,
    deps = [
        "//cyber",
        "//component/sender/proto:sender_proto",

    ],
)

apollo_package()

cpplint()