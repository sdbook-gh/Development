add_rules("mode.debug", "mode.release")

-- 定义一个名为 "dotnet" 的规则
rule("dotnet")
    -- 设置该规则支持的文件扩展名
    set_extensions(".csproj", ".sln")
    -- 定义构建文件时的行为
    on_build_file(function (target, sourcefile, opt)
        mode = "Release"
        if is_mode("debug") then
            mode = "Debug"
        end
        print("Building C# project: %s mode: %s sourcefile: %s", target:name(), mode, sourcefile)
        -- 使用 dotnet build 命令构建项目
        os.exec("dotnet build -c %s %s", mode, sourcefile)
    end)
    -- 定义清理文件时的行为
    on_clean(function (target)
        mode = "Release"
        if is_mode("debug") then
            mode = "Debug"
        end
        print("Cleaning C# project: %s mode: %s", target:name(), mode)
        -- 使用 dotnet clean 命令清理项目
        os.exec("dotnet clean -c %s", mode)
    end)
    on_link(function (target)
    end)

target("test_animal")
    -- 设置目标类型为二进制文件
    set_kind("binary")
    add_rules("dotnet")
    -- 添加 C# 项目文件，并应用 "dotnet" 规则
    add_files("test_animal.csproj")
