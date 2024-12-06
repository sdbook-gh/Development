load("//tools:apollo_package.bzl", "apollo_package")
load("//tools/proto:proto.bzl", "proto_library")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "rtsp_camera_trigger_proto",
    srcs = ["rtsp_camera_trigger.proto"],
)

apollo_package()

cpplint()
