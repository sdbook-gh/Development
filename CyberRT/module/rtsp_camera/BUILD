load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_cc_binary", "apollo_package", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "rtsp_camera_files",
    srcs = glob([
        "dag/**",
        "launch/**",
        "conf/**",
    ]),
)

apollo_component(
    name = "librtsp_camera_component.so",
    srcs = [
        "rtsp_camera_component.cc",
    ],
    hdrs = [
        "rtsp_camera_component.h",
    ],
    linkstatic = True,
    deps = [
        "//cyber",
        "//modules/common_msgs/sensor_msgs:sensor_image_cc_proto",
        "//module/rtsp_camera_trigger/proto:rtsp_camera_trigger_proto",
    ],
)

apollo_package()

cpplint()
