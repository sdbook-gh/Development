# RK3588 USB 插入死机 / 无法发现设备：排查与修复

产品：`rk3588_t` / 板卡 `rd-box-rk3588`  
口：Type-A OTG（`usbdrd_dwc3_0` / `fc000000.usb`，`dr_mode=otg`，extcon=`u2phy0`）  
现象（两个，叠加）：

1. **先上电再插 USB**：整机像死机（其实是关机）。先插 USB 再上电则 ADB 正常。
2. 规避死机之后：**后插 USB 仍枚举不上**（Windows 发现不了设备）。带线上电仍可枚举。

---

## 1. 排查过程

### 1.1 环境

- 源码：`/home/jx/SourceCode/Android_13`。
- 远端：`ssh -p 2201 shenda@192.168.66.112`，`ADB=/mnt/d/devd/platform-tools/adb.exe`。
- 后插线会死时没有 UART / 网口备用入口，只能靠 `/data` 落盘 + 带线上电把数据拉回来。
- 死机规避后可用 WiFi ADB：`persist.internet_adb_enable=1`，设备 `192.168.66.144:5555`。

### 1.2 分层思路

先分清是 PC 完全没枚举，还是枚举了但不是 adb；再分清是角色/充电器检测，还是用户态关机：

PC 枚举（VID 2207）→ PHY bvalid / 充电器检测（SDP/CDP/DCP）→ DWC3 dr_role → UDC/gadget → adbd  
同时对照：带线上电（好） vs 后插线（坏）

| 阶段 | 做法 | 结果 |
|------|------|------|
| 只读对照 | 带线上电采 sysfs | `dwc3=device`，UDC configured，CDP，adb 在 |
| 采集镜像 | 内核 USBDBG + `usb_debug.sh` 落 `/data/usb_debug` | 死前约 1s 内的 kmsg/logcat/usbdbg 能留下来 |
| 好/坏对比 | `NEXT_LABEL` 标记 `good` / `lateplug` | 坏场景不是 kernel panic，是 **关机** |
| 规避死机 | `rk805_pwrkey` 的 KEY_POWER 映成 F21 | 后插线 **不再关机** |
| 强制 peripheral | WiFi adb：`echo peripheral > otg_mode` | USB adb **立刻出现**，线/gadget 没坏 |
| 修充电器路径 | DCD 超时当 SDP；DCP 也 go peripheral | 后插线可枚举 |

### 1.3 编译踩坑：`make bootimage` 不含内核源码改动

Rockchip 用 `TARGET_PREBUILT_KERNEL = kernel-5.10/arch/arm64/boot/Image`。  
`make bootimage` 只把已有 Image 打进 boot，**不编 kernel**。  
必须 `./build.sh -CK`，再用 `strings boot.img | grep USBDBG-` 确认标记进了镜像。

---

## 2. 运行过的命令

### 2.1 活着时的 USB 快照

```bash
OTG=/sys/devices/platform/fd5d0000.syscon/fd5d0000.syscon:usb2-phy@0/otg_mode
DWC3=/sys/kernel/debug/usb/fc000000.usb/mode

cat $OTG
cat $DWC3
cat /sys/class/udc/fc000000.usb/state
cat /sys/class/udc/fc000000.usb/current_speed
cat /config/usb_gadget/g1/UDC
cat /sys/class/extcon/extcon0/state
getprop sys.usb.config
getprop sys.usb.state
dmesg | grep USBDBG
```

### 2.2 采集服务

设备：`/data/usb_debug/`（状态变化立刻 `sync`）

- `LAST_PHASE`：`UNPLUGGED` / `BOOT_WITH_USB_SUCCESS` / `LATE_PLUG`
- `${LABEL}_${STAMP}/`：`kmsg.txt` `logcat.txt` `usbdbg.txt` `events.txt` `getevent.txt` `poll.latest`
- 复现前：`echo lateplug > /data/usb_debug/NEXT_LABEL`

后插线死亡场景操作：断电拔线 → 上电 ≥90s → 插 USB → 若死则 **保持插线** 再上电（回到好场景）拉数据。

### 2.3 强制切 peripheral（WiFi adb 对照）

```bash
echo peripheral > $OTG
echo device > $DWC3
# 随后 udc=configured / high-speed，USB adb 出现
```

### 2.4 编译与确认内核进了镜像

```bash
cd /home/jx/SourceCode/Android_13
./build.sh -CK -J$(nproc)
strings out/target/product/rk3588_t/boot.img | grep USBDBG-20260901-v2
```

---

## 3. 分析出的异常

### 3.1 「死机」其实是伪造 POWER 长按关机

死亡轮（uptime）：

| 时间 | 事件 |
|------|------|
| 4.35s | **没插线** 就 `bvalid irq`，`utmi_bvalid=1` |
| 5.02s | 充电器误判 **DCP** → `dcp cable, stay idle (no gadget)` |
| 5s~108s | 每秒一次 `vbus_attach → DCP → stay idle` |
| **104.3s** | 插线瞬间：`rk805 pwrkey` 发出 **KEY_POWER DOWN**（无有效短按释放） |
| 104.8s | `powerLongPress`，`mLongPressOnPowerBehavior=3`（SHUT_OFF_NO_CONFIRM） |
| 105.3s | `sys.powerctl='shutdown,userrequested'` |
| 108.5s | init 关机 → 用户看到「死机」 |

好场景（带线上电）：只两次 `vbus_attach`，判 **CDP** → `go peripheral` → 枚举成功；logcat **0** 次 power 键。

键源：`/dev/input/event0`（`rk805 pwrkey`）。驱动无防抖，下降沿直接 `KEY_POWER 1`。`adc-keys` 键表没有 116。rk3588_t 未 overlay `config_longPressOnPowerBehavior`，用的是 3。

规避（已验证不死）：设备上把该输入设备的 **scan 116 映成 F21**。  
注意：VID/PID 都是 0，`Vendor_XXXX_Product_XXXX.kl` **不会被查找**；只推 `rk805_pwrkey.kl` 缺 `.kcm` 会回落到 `Generic.kl`。最终是改生效的 keylayout（设备上的 `Generic.kl`：`key 116 F21`），或 `rk805_pwrkey.kl` + `.kcm` + `.idc` 三件套。

驱动防抖仍未做，这是 Android 层规避。

### 3.2 无法发现设备：误判 DCP 后 stay idle

上电不插线：bvalid 浮空为 1 → DCD 超时被标成 DCP → 每秒 stay idle。  
之后再插 PC 线 **不会重新当成 SDP/CDP**，DWC3 停在 UNKNOWN，UDC `not attached`。

WiFi 上强制 `otg_mode=peripheral` 立刻枚举成功 → gadget 和线都好，卡在 PHY 把口当成充电器。

带线上电走 CDP 所以正常。

---

## 4. 最后修改的文件

| 文件 | 改动 |
|------|------|
| `kernel-5.10/drivers/phy/rockchip/phy-rockchip-inno-usb2.c` | USBDBG 宏（现 `USBDBG-20260901-v2`）；DCD 超时改为 `POWER_SUPPLY_TYPE_USB`（SDP），不再当 DCP；DCP 分支改为 **go peripheral**，不再 stay idle |
| `kernel-5.10/drivers/usb/dwc3/core.c` | `__dwc3_set_mode` 用 `dev_err` 打 dr_role |
| `kernel-5.10/drivers/input/misc/rk805-pwrkey.c` | fall/rise 中断各一行 USBDBG（观测，不改行为） |
| `device/rockchip/common/rootdir/usb_debug.sh` | 开机采集；状态分类；`getevent` 落盘；变化立刻 sync |
| `device/rockchip/common/rootdir/usb_debug.rc` | init 服务 |
| `device/rockchip/common/rootdir/usb_debug.version` | 版本标记 |
| 设备 `/system/usr/keylayout/Generic.kl` 或 `rk805_pwrkey.{kl,kcm,idc}` | `key 116 F21`，规避伪 POWER 关机（源码树里的 Generic.kl 未改） |

---

## 5. 验证要点

1. `dmesg | grep 'USBDBG: mark='` 能看到当前标记（现 `USBDBG-20260901-v2`），证明刷到的是新内核，不是旧 Image。
2. 带线上电：`cdp/sdp/... go peripheral`，USB adb 正常。
3. 后插线：不死；Windows 能出 adb。不应再出现每秒 `stay idle (no gadget)`。
4. `dumpsys input` 里 `rk805 pwrkey` 的 KeyLayout 已把 116 映成非 POWER。
5. `init.svc.usb_debug` = `running`；`/data/usb_debug/LAST_PHASE` 与场景一致。
