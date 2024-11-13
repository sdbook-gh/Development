def _run_cmake_impl(ctx):
    cmake_project_path = ctx.attr.cmake_project_path
    out_file = ctx.outputs.output
    output_dir = ctx.actions.declare_directory(ctx.label.name)

    # out_file = ctx.actions.declare_file(ctx.label.name + "/_result")
    ctx.actions.run_shell(
        outputs = [out_file, output_dir],
        progress_message = "building...",
        command = "touch {out_file} && export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:$PATH && export BASE_DIR=$(pwd) && echo $BASE_DIR && mkdir \"$BASE_DIR/_build\" && cd \"$BASE_DIR/_build\" && cmake -DOUTPUT_DIR=\"$BASE_DIR/{output_path}\" \"{cmake_project_path}\" && cmake --build \"$BASE_DIR/_build\" --target install".format(out_file = out_file.path, output_path = output_dir.path, cmake_project_path = cmake_project_path),
    )

run_cmake = rule(
    implementation = _run_cmake_impl,
    attrs = {
        "cmake_project_path": attr.string(
            mandatory = True,
        ),
        "cmake_macro_list": attr.string_list(
        ),
        "output": attr.output(
        ),
    },
)

def _run_cmake2_impl(ctx):
    cmake_project_path = ctx.attr.cmake_project_path
    output_dir = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run_shell(
        inputs = [],
        outputs = [output_dir],
        command = "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:$PATH && export BASE_DIR=$(pwd) && mkdir \"$BASE_DIR/_build\" && cd \"$BASE_DIR/_build\" && cmake -DOUTPUT_DIR=\"$BASE_DIR/{output_path}\" \"{cmake_project_path}\" && cmake --build \"$BASE_DIR/_build\" --target install".format(output_path = output_dir.path, cmake_project_path = cmake_project_path),
    )
    return [DefaultInfo(files = depset([output_dir]))]

run_cmake2 = rule(
    implementation = _run_cmake2_impl,
    attrs = {
        "cmake_project_path": attr.string(
            mandatory = True,
        ),
        "cmake_macro_list": attr.string_list(
        ),
    },
)
