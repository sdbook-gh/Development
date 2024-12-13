load("//tools:apollo_package.bzl", "apollo_package", "apollo_cc_library", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

HESAI_COPTS = ['-DMODULE_NAME=\\"hesai-lidar\\"']

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
    name = "libhesai_component.so",
    srcs = [
        "src/hesai_component.cpp",
    ],
    hdrs = [
        "src/hesai_component.h",
    ],
    deps = [
        "//cyber",
        "//modules/common_msgs/sensor_msgs:pointcloud_cc_proto",
        "//module/hesai_lidar/proto:hesai_config_proto",
        "//module/hesai_lidar/proto:hesai_proto",
    ],
    copts = HESAI_COPTS + [
        "-std=c++17",
        "-fopenmp",
    ],
    linkopts = [
        "-lgomp", "-lssl", "-pthread", "-lm", "-lhesai2-driver",
    ]
)

apollo_package()
cpplint()
