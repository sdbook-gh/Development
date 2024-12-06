load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_cc_binary", "apollo_package", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "rtsp_camera_trigger_files",
    srcs = glob([
        "conf/**",
    ]),
)

apollo_cc_binary(
    name = "rtsp_camera_trigger",
    srcs = ["rtsp_camera_trigger.cc"],
    deps = [
        "//cyber",
        "//module/rtsp_camera_trigger/proto:rtsp_camera_trigger_proto",
    ],
)

apollo_package()

cpplint()
