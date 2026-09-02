# 从 U-Boot fastboot 切到 rktools 可识别（Loader）

产品：`rk3588_t` / 板卡 `rd-box-rk3588`  
现象：`fastboot devices` 能看到设备，RKDevTool / rktools **发现不了、不能烧录**。

---

## 结论

rktools 只认 **RockUSB Loader / Maskrom**（VID `2207`），不认 Google fastboot（VID `18D1`）。

本板已验证路径（U-Boot fastboot 里没有直接进 Loader 的 oem 命令）：

```
fastboot reboot-recovery  →  等 adb 出现 recovery  →  adb reboot loader
```

成功标志：Windows 枚举 `Rockusb Device`，`USB\VID_2207&PID_350B\...`（RK3588 Loader）。  
打开 `D:\devd\RKDevTool_Release_v3.31\RKDevTool.exe`，应显示 **发现一台 LOADER 设备**。

---

## 1. 环境

- 编译机 `192.168.66.130`；USB 在远端 `ssh -p 2201 shenda@192.168.66.112`。
- **必须**用 Windows 平台工具（别名在非交互 SSH 里不生效，写全路径）：

```bash
ADB=/mnt/d/devd/platform-tools/adb.exe
FB=/mnt/d/devd/platform-tools/fastboot.exe
```

- rktools：`D:\devd\RKDevTool_Release_v3.31\RKDevTool.exe`
- Linux `/usr/bin/adb`、WSL 里的 `fastboot` 都看不到这台 USB 设备。

---

## 2. 为什么 rktools 看不到

| 模式 | USB | 谁能看见 |
|------|-----|----------|
| Android / recovery ADB | `18D1:4EE0` 一类 | `adb devices` |
| U-Boot Google fastboot | `18D1:D00D` | `fastboot devices` |
| **Rockchip Loader** | **`2207:350B`** | **RKDevTool / upgrade_tool** |
| Maskrom | `2207:350A` | RKDevTool（MASKROM） |

`18D1:D00D` 在 Windows 上常显示成 “Android ADB Interface”，容易误判。以 VID/PID 为准，不要看 FriendlyName。

U-Boot 2017.09 进 fastboot 后，`fastboot reboot` 仍会回到 U-Boot fastboot；`fastboot continue` 会尝试启内核，约 15s 后又回到 fastboot。这两条 **都进不了 RockUSB**。

本树 `f_fastboot.c` 的 oem 只有 unlock/format/AVB 等，**没有** `oem download` / `reboot loader`。

内核 DTS（`rk3588s.dtsi` reboot-mode）里：

- `reboot loader` / `reboot bootloader` → `BOOT_BL_DOWNLOAD` → U-Boot `download`（RockUSB）
- `reboot fastboot` → `BOOT_FASTBOOT` → U-Boot `fastboot usb 0`

所以要从 **已经跑起来的 recovery/Android** 发 `adb reboot loader`，不能从 U-Boot fastboot 里 `fastboot reboot`。

---

## 3. 操作步骤（已跑通）

在 `192.168.66.112` 上：

```bash
ADB=/mnt/d/devd/platform-tools/adb.exe
FB=/mnt/d/devd/platform-tools/fastboot.exe

# 0) 确认当前是 U-Boot fastboot
$FB devices -l
# 期望：54c970a9db96e5e6    fastboot
# USB：VID_18D1&PID_D00D

# 1) 先切到 recovery（这条在 U-Boot fastboot 下是通的）
$FB reboot-recovery

# 2) 等 adb 上线（大约 15s）
$ADB devices -l
# 期望：54c970a9db96e5e6    recovery

# 3) 再进 Loader
$ADB reboot loader
```

若 recovery 上 `adb reboot loader` 报 `device not found`，先 `$ADB root`，等几秒再执行（`wait-for-device` 在 recovery 上容易超时，用 `sleep 3` 即可）。

### 3.1 确认已进 Loader

PowerShell：

```powershell
Get-PnpDevice -PresentOnly |
  Where-Object { $_.InstanceId -like 'USB\VID_2207*' } |
  Select-Object Status, FriendlyName, InstanceId
```

期望：

```
OK  Rockusb Device  USB\VID_2207&PID_350B\54C970A9DB96E5E6
```

此时 `adb devices`、`fastboot devices` 都应是空的，这是正常现象。

然后打开 RKDevTool。设备会停在 Loader 等烧录，不会自己退回 fastboot。

### 3.2 已经在 recovery，只需进 Loader

```bash
$ADB devices -l          # 已是 recovery 则可跳过 reboot-recovery
$ADB reboot loader
```

### 3.3 已经在 Android 系统

```bash
$ADB reboot loader
```

若系统起不来、只有 U-Boot fastboot，仍走第 3 节两步。

---

## 4. 不要用的做法

| 做法 | 结果 |
|------|------|
| 只 `fastboot reboot` | 还在 U-Boot fastboot，`18D1:D00D` |
| `fastboot continue` | 内核跑约 15s 又回 fastboot |
| `fastboot reboot-bootloader` | U-Boot 写的是 `BOOT_FASTBOOT`，还是 Google fastboot |
| 对 `18D1:D00D` 开 RKDevTool | ScanLog 里 `get serial string failed, err=0x1f` |
| 按 volume 进下载 | 那是 Loader/Maskrom 的按键路径；USB 已在 fastboot 时不必靠按键 |

需要 Maskrom（`2207:350A`）时：断电，按住 recovery/下载键再上电（U-Boot 里 `rockchip_dnl_key_pressed` + VBUS → `download`）。日常烧录用 Loader `350B` 即可。

---
