"""Custom rule to create a d8_compat_dx wrapper that fixes the --min_sdk_version flag.

Bazel 7.2.1's OptionsParser does not convert underscores to dashes, but
android_binary passes --min_sdk_version (underscores) while CompatDx expects
--min-sdk-version (dashes). This wrapper rewrites the flag before delegating.
"""

def _d8_wrapper_impl(ctx):
    wrapper = ctx.actions.declare_file(ctx.label.name + "_wrapper.sh")
    ctx.actions.expand_template(
        template = ctx.file._template,
        output = wrapper,
        substitutions = {
            "@D8_COMPAT_DX@": ctx.executable._d8_compat_dx.path,
        },
        is_executable = True,
    )
    d8_info = ctx.attr._d8_compat_dx[DefaultInfo]
    runfiles = ctx.runfiles(files = [ctx.executable._d8_compat_dx])
    runfiles = runfiles.merge(d8_info.default_runfiles)
    return [
        DefaultInfo(
            executable = wrapper,
            runfiles = runfiles,
        ),
    ]

d8_wrapper = rule(
    implementation = _d8_wrapper_impl,
    attrs = {
        "_template": attr.label(
            allow_single_file = True,
            default = Label("//tools:d8_wrapper.sh.tpl"),
        ),
        "_d8_compat_dx": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@androidsdk//:d8_compat_dx"),
        ),
    },
    executable = True,
)
