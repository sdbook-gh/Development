load("//tools:apollo_package.bzl", "apollo_package", "apollo_cc_library", "apollo_component")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

SEYOND_COPTS = ['-DMODULE_NAME=\\"seyond\\"']

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
    name = "libseyond_lidar.so",
    srcs = [
        "src/seyond_lidar.cpp",
        "src/seyond_driver.cpp",
    ],
    hdrs = [
        "src/seyond_lidar.h",
        "src/seyond_driver.h",
    ],
    copts = SEYOND_COPTS + ["-std=c++17"],
    deps = [
        "//cyber",
        "//modules/common_msgs/sensor_msgs:pointcloud_cc_proto",
        "//module/seyond_lidar/proto:seyond_config_proto",
        "//module/seyond_lidar/proto:seyond_proto",
    ],
    linkopts = [
        "-linnoclientsdk",
    ]
)

apollo_package()
cpplint()
