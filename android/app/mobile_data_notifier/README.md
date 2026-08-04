# 移动数据通知器 (mobile_data_notifier)

在手机屏幕顶部用悬浮窗显示**移动数据开关状态**与**已用流量**，可一键跳转关闭移动数据；支持**开机自启**、**前台服务保活**、**定时关闭/开启提醒**（到点带声音与震动）。

> ⚠️ **未 Root 设备限制**：现代 Android 不允许普通第三方 App 真正开关移动数据。
> 因此"关闭流量"按钮与定时提醒均**跳转到系统数据使用设置页**，由用户手动关闭/开启。
> 读取开关状态、读取流量、悬浮窗显示、开机自启、保活、到点声音+震动提醒等功能均正常。

## 技术栈

- Kotlin + Bazel 7.2.1 + rules_kotlin 1.9.0
- 仅使用 Android 框架 API（不引入 AndroidX，避免 Maven 依赖；`maven.google.com` 不可达时仍可编译）
- compileSdk 35 / targetSdk 34 / minSdk 26（Android 8.0+）
- 复用本机 Android SDK：`/mnt/d/devd/android/sdk-linux`

## 目录结构

```
mobile_data_notifier/
├── .bazelrc .gitignore bazel(wrapper) WORKSPACE BUILD.bazel README.md
├── tools/  BUILD.bazel d8_wrapper.bzl d8_wrapper.sh.tpl   # d8 下划线→连字符补丁
└── src/main/
    ├── AndroidManifest.xml
    ├── res/{values,layout,drawable}/...
    └── java/com/example/mobiledatanotifier/
        MainActivity.kt        # 权限、状态展示、悬浮窗/保活控制、定时时段管理
        OverlayService.kt      # 前台服务(保活) + 顶部悬浮窗 + 网络监听
        BootReceiver.kt        # 开机自启
        ScheduleReceiver.kt    # 到点提醒(声音+震动)
        ScheduleManager.kt     # AlarmManager 排程
        Prefs.kt               # SharedPreferences 封装
        DataMonitor.kt         # 读数据状态/流量
        PermUtil.kt            # 各类特殊权限跳转
        TimePeriod.kt          # 时段数据模型
```

## 编译

输出目录已通过 `./bazel` 包装脚本设为项目内 `.bazel_output/`。

```bash
cd /mnt/d/personal/github/Development/android/app/mobile_data_notifier
./bazel build //:mobile_data_notifier
```

产物：`bazel-bin/mobile_data_notifier.apk`

## 安装到设备

```bash
adb devices                       # 确认设备已连接
adb install -r bazel-bin/mobile_data_notifier.apk
```

## 首次使用（权限授予顺序）

1. 打开 App → 系统会请求 **电话权限**（读数据开关状态）、**通知权限**。
2. 点 **"授予悬浮窗权限"** → 在系统设置里开启"显示在其他应用上层"。
3. 点 **"加入电池优化白名单"** → 允许（保活）。
4. （如定时提醒不准）点电池状态文字 → 授予 **精确闹钟** 权限。
5. 点 **"启动保活服务"** → 顶部出现悬浮窗。
6. 在"定时关闭"里 **"+ 添加时段"**，点开始/结束时间用时间选择器设定（如 23:00 → 07:00），开关启用。

## 功能说明

- **悬浮窗**：显示移动数据开关状态、开机以来流量；可拖动；"关闭流量"按钮跳转系统数据使用设置；"×"隐藏（服务仍在后台，可从前台通知"显示悬浮窗"恢复）。
- **保活**：前台服务 `foregroundServiceType="specialUse"` + 常驻低优先级通知（静音）+ `START_STICKY` + 电池白名单；开机自启。
- **定时提醒**：到开始时间弹"请关闭移动数据"高优先级通知（声音+震动+横幅），附"关闭数据"按钮跳转设置；到结束时间弹"可重新开启"提醒。
- **流量统计**：开机以来累计 + 本次统计（可重置）。

## 备注

- 通知小图标为矢量 `ic_notify.xml`；应用图标为矢量 `ic_launcher.xml`。
- 若需**真正自动开关**移动数据，需设备 Root（改用 `su -c "svc data disable/enable"`）或 AccessibilityService 方案，当前未实现。
