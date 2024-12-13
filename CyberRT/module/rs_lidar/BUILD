load("//tools:apollo_package.bzl", "apollo_package", "apollo_cc_library", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

ROBOSENSE_COPTS = ['-DMODULE_NAME=\\"rs-lidar\\"']

filegroup(
    name = "runtime_data",
    srcs = glob([
        "conf/*.txt",
        "conf/*.conf",
        "dag/*.dag",
        "launch/*.launch",
    ]),
)

apollo_component(
    name = "librs_component.so",
    srcs = [
        "src/rs_component.cpp",
    ],
    hdrs = [
        "src/rs_component.h",
    ],
    deps = [
        "//cyber",
        "//modules/common_msgs/sensor_msgs:pointcloud_cc_proto",
        "//module/rs_lidar/proto:rs_config_proto",
        "//module/rs_lidar/proto:rs_proto",
    ],
    copts = ROBOSENSE_COPTS + [
        "-std=c++17",
    ],
    linkopts = [
        "-lpcap",
    ]
)

apollo_package()
cpplint()
