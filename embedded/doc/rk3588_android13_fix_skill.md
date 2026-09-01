---
name: rk3588-connectivity
description: Debug and fix rk3588_t (rd-box-rk3588).
---

# RK3588 连接问题排查

产品默认：`rk3588_t` / `rd-box-rk3588`，源码 `/home/jx/SourceCode/Android_13`。

## 访问约定

- 编译机 `192.168.66.130`；板子经 `ssh -p 2201 shenda@192.168.66.112`。
- **必须**用 `/mnt/d/devd/platform-tools/adb.exe`（及 `fastboot.exe`）。Linux `/usr/bin/adb` 看不到这台 USB 设备。
- 用户要求询问-执行时：只读可连着做；拔插线、改 GPIO/PHY、重启、烧录、改代码前先问。
- 交互别名不生效。非交互 SSH 写全路径，不要依赖 `~/.bashrc` 的 `alias adb=`。

## 总流程

1. 用现有 log/sysfs **分层定位**，先不改代码加日志。
2. 做能翻转一层的对照实验（开关间隔 vs 真掉电；对照板同线；带线上电 vs 后插线）。
3. 改动后用 **设备上能读到的标记** 证明镜像真刷进去了，不要信 `ro.bootimage.build.date`（增量编译常不更新）。

## 编译与烧录（必查）

| 要验证 | 正确做法 | 常见假象 |
|--------|----------|----------|
| 内核 C 改动 | `./build.sh -CK`，`strings boot.img \| grep` 标记 | `make bootimage` 只打包旧 `Image` |
| DTS 进内核 | 换 resource 里的 `rk-kernel.dtb`，或完整 -CK | 只换 `--dtb` 槽；`/proc/device-tree` 仍旧 |
| 已刷上 | 设备 `dmesg` 标记、`/proc/version`、boot 分区 MD5 | `ro.bootimage.build.date` 来自缓存 |
| fastboot | Windows `fastboot.exe`；virtiofs 路径用 `wslpath` | WSL `fastboot` 看不到 USB |

烧 boot 后若停在 bootloader，再 `fastboot reboot` 一次。vendor 脚本可用 `adb root` + `remount` push，不必每次刷 vendor。

无线备用：`setprop persist.internet_adb_enable 1`（重启仍开 5555）。

## 不要做的事

- 用 `make bootimage` 验证内核补丁。
