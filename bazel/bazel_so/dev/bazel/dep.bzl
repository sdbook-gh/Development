load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
def clean_dep(dep):
    return str(Label(dep))
def third_repo():
    new_local_repository(
        name = "third",
        build_file = clean_dep("//dev/bazel:test_so.BUILD"),
        path = "/home/shenda/bazel/bazel_test/third",
    )

def init_deps():
    third_repo()
