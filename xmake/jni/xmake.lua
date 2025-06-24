-- 设置项目
set_project("MyAndroidApp")
set_version("1.0")

-- 添加 Android NDK 构建规则
add_rules("mode.debug", "mode.release")
-- add_rules("platform.android.ndk")

-- 配置 Android 目标
target("native-lib")
    set_kind("shared")  -- 生成 .so 动态库
    add_files("src/*.cpp")
    
    -- Android 特定配置
    -- set_plat("android")
    -- set_arch("arm64-v8a")  -- 可选: armeabi-v7a, x86, x86_64
    
    -- NDK 设置 (自动检测环境变量 ANDROID_NDK_HOME)
    -- 或手动指定: set_toolchains("ndk", {ndk = "/path/to/ndk"})
    
    -- 头文件搜索路径
    -- add_includedirs("$(env(ANDROID_NDK_HOME)/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include")
    
    -- 链接系统库
    add_syslinks("log")  -- 支持 __android_log_print
    
    -- C++ 标准
    add_cxxflags("-std=c++17", "-fexceptions", "-frtti")
    
    -- 禁用异常和 RTTI（如需）
    -- add_cxxflags("-fno-exceptions", "-fno-rtti")

    add_ldflags("-lc++_shared")
