load("//tools:apollo_package.bzl", "apollo_package")
load("//tools/proto:proto.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "rs_config_proto",
    srcs = ["rs_config.proto"],
)

proto_library(
    name = "rs_proto",
    srcs = ["rs.proto"],
    deps = [
        "//modules/common_msgs/basic_msgs:header_proto",
    ],
)

apollo_package()
