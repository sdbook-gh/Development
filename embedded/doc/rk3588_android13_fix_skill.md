---
name: rk3588-connectivity
description: Debug and fix rk3588_t (rd-box-rk3588) WiFi scan, ethernet IP/link, and USB OTG late-plug issues on Android 13. Use when WiFi cannot discover networks, eth0 has no IP or cannot reach gigabit, USB insert freezes/shuts down the board, or adb cannot enumerate after power-on-then-plug.
---

# RK3588 连接问题排查

产品默认：`rk3588_t` / `rd-box-rk3588`，源码 `/home/jx/SourceCode/Android_13`。

案例全文（只在对应该问题时再读）：

- [fix_wifi.md](fix_wifi.md) — WiFi 二次打开搜不到
- [fix_eth.md](fix_eth.md) — 网线无 IP / 达不到千兆
- [fix_usb.md](fix_usb.md) — USB 后插死机 / 枚举失败

## 访问约定

- 编译机 `192.168.66.130`；板子经 `ssh -p 2201 shenda@192.168.66.112`。
- **必须**用 `/mnt/d/devd/platform-tools/adb.exe`（及 `fastboot.exe`）。Linux `/usr/bin/adb` 看不到这台 USB 设备。
- 用户要求询问-执行时：只读可连着做；拔插线、改 GPIO/PHY、重启、烧录、改代码前先问。
- 交互别名不生效。非交互 SSH 写全路径，不要依赖 `~/.bashrc` 的 `alias adb=`。

## 总流程

1. 用现有 log/sysfs **分层定位**，先不改代码加日志。
2. 做能翻转一层的对照实验（开关间隔 vs 真掉电；对照板同线；带线上电 vs 后插线）。
3. 改动后用 **设备上能读到的标记** 证明镜像真刷进去了，不要信 `ro.bootimage.build.date`（增量编译常不更新）。
4. 写进案例 md 的同类问题，按对应文档里的文件改，不要另起一套。

```
现象落在哪一层？
  WiFi 搜不到     → 第二节
  eth0 无 IP/速率 → 第三节
  USB 死机/不枚举 → 第四节
  编译烧录不对    → 第五节
```

## WiFi 搜不到

路径：Settings → WifiService → wificond/wpa_supplicant → HAL/bcmdhd → AP6398S + SDIO。

已修根因（AP6398S）：

- 关 WiFi **没拉 REG_ON**：DTS 写了 `WIFI,vbat_gpio`，rfkill 认 `WIFI,poweren_gpio` → GPIO0_C4 一直高。第二次 `mmc_hw_reset` 报 **-110**。
- 固件路径写成不存在的 `fw_bcmdhd.bin`/`nvram.txt`，应对 `fw_bcm4359c0_ag.bin` + `nvram_ap6398s.txt`。

对照：设置里空等再开（无效）vs 写 `/sys/class/rkwifi/wifi_bt_vbat` 掉电并 `ctl.restart vendor.wifi_hal_legacy`（有效）。HAL 失败后会停在 Idle，只脉冲电源不够。

改这些文件（不要改 Settings/WifiService）：

- `kernel-5.10/arch/arm64/boot/dts/rockchip/rk3588/rp-wifi-bt-ap6398s-rk3588s.dtsi`：`WIFI,poweren_gpio`（不要和 bcmdhd 抢同一脚）
- `kernel-5.10/net/rfkill/rfkill-wlan.c`：拉脚后 `msleep(200)`
- `device/rockchip/common/init.connectivity.rc`：正确 firmware/nvram

验证：关/开三轮仍能扫；dmesg 有 `wifi shut/turn on power`，**没有** `mmc2 ... -110`。

## 网线无 IP / 非千兆

路径：线/hub → PHY 自协商 → RGMII delay → Link Up → EthernetTracker/DHCP → IPv4。

- 先读 `speed`/`operstate`/`carrier`、`ip -br addr`、`dmesg` 的 `Link is`/`TX delay`、`dumpsys ethernet`。
- 同 hub 同网线对照板能千兆 → 排除线。本板 RTL8211F 只宣告/强制 1000 仍 down → **1000Base-T 训练失败**，不是 delay 能修的。
- `tx_delay` 只影响 RGMII。改 DTS 后必须看 `/proc/device-tree/ethernet@fe1c0000/tx_delay` 和 dmesg `TX delay(0x..)`。
- **RK3588 内核 DTB 在 boot.img 的 `second`（resource.img → `rk-kernel.dtb`）**。只换 header v2 `--dtb` 槽，运行时仍是旧树。
- Link Up 但没 IPv4：看 EthernetTracker/DHCP 是否在新 carrier 上重试。`dhcpcd_eth0` 与 `start_ethernet` 默认 disabled。不要用静态 `ifconfig` 当产品方案。

源码里 `rp-eth-gmac1.dtsi` 的 `tx_delay=0x45` 曾用于对照，**当时板上仍是 0x44**。

## USB 后插死机 / 发现不了设备

口：Type-A OTG，`fc000000.usb`，extcon=`u2phy0`。先插线再上电正常；先上电再插线出问题。

两个独立问题，不要合成一个补丁糊弄：

**A. 「死机」= 伪 POWER 长按关机**

- 插线瞬变 → `rk805 pwrkey` KEY_POWER DOWN → 500ms → `mLongPressOnPowerBehavior=3` → `shutdown,userrequested`。
- 规避：该设备 scan 116 → F21。VID/PID=0 **不会**走 `Vendor_*` kl；单文件 `rk805_pwrkey.kl` 缺 `.kcm` 会回落 `Generic.kl`。要用 kl+kcm+idc 或改已生效的 Generic.kl。驱动防抖仍待做。

**B. 枚举失败 = 误判 DCP 后 stay idle**

- 不插线 `utmi_bvalid=1` → DCD 超时当 DCP → 每秒 `stay idle (no gadget)` → 后插 PC 不再走 gadget。
- 对照：WiFi adb 下 `echo peripheral > otg_mode` 立刻枚举 → 线/gadget 没坏。
- 已修：`phy-rockchip-inno-usb2.c` — DCD 超时当 SDP；DCP 也 **go peripheral**。标记 `USBDBG-20260901-v2`。

无备用通道时：用 `usb_debug.sh` 写 `/data/usb_debug`，变化立刻 sync。复现前写 `NEXT_LABEL`。死后保持插线再上电拉数据。好/坏场景对比 `usbdbg.txt` + logcat power 键 + `getevent`。

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

- 未分层就改 Settings / WifiService / 乱改 `tx_delay` 当千兆开关。
- 给 bcmdhd 和 rfkill 申请同一 REG_ON GPIO。
- 用 `make bootimage` 验证内核补丁。
- 把 USB 死机当 panic，不去对 logcat 的 `ShutdownThread` / `intercept_power`。
- 假设 DCP stay idle 只是充电器，后插 PC 就会自动切 gadget。
