# RK3588 二次打开 WiFi 搜不到：排查与修复

产品：`rk3588_t` / 板卡 `rd-box-rk3588`  
模组：AP6398S（BCM4359，SDIO + bcmdhd）  
现象：每次重启后，系统 Settings **第一次打开 WiFi 能搜到网络，关掉再打开就搜不到**。

---

## 1. 排查过程

### 1.1 环境

- 源码与编译在本机 `192.168.66.130`（`/home/jx/SourceCode/Android_13`）。
- Android 板挂在远端 `192.168.66.112`，登录：`ssh -p 2201 shenda@192.168.66.112`。
- 约定：对设备执行任何操作前先询问；抓 log 前再单独问一次。

### 1.2 分层思路

不先改代码加日志，用现有 logcat / dmesg / 状态快照判断失败落在哪一层：

Settings → WifiService → wificond / wpa_supplicant → wifi HAL / bcmdhd → AP6398S + SDIO

对照实验：

| 实验 | 做法 | 结果 |
|------|------|------|
| A | 设置里关 WiFi，等 10 秒再开 | 仍搜不到 |
| B | 手动拉 GPIO0_C4（VBAT/REG_ON）低 1 秒再拉高，并重启 wifi HAL | 驱动 `wifi_on Success`，设置能搜到 |

### 1.3 adb 踩坑

远端交互 shell 能看到设备，本机非交互 `ssh ... adb devices` 一开始是空的。

原因：`~/.bashrc` 里有

```bash
alias adb='/mnt/d/devd/platform-tools/adb.exe'
```

别名只在交互 shell 生效。非交互走的是 Linux `/usr/bin/adb`，看不到 Windows 侧 USB。  
后续一律使用：`/mnt/d/devd/platform-tools/adb.exe`。设备序列号：`db95636af0bbb625`。

---

## 2. 运行过的命令

以下命令均通过 `ssh -p 2201 shenda@192.168.66.112` 在远端执行（本机用 `SSH_ASKPASS` 传密码，不用 Python）。`ADB` 指 `/mnt/d/devd/platform-tools/adb.exe`。

### 2.1 连通性

```bash
ssh -p 2201 shenda@192.168.66.112 'adb devices -l'          # Linux adb，列表为空
ssh -tt ... "$ADB devices -l"                                 # 看到 db95636af0bbb625 device
```

### 2.2 抓 log（复现前清空）

```bash
$ADB root
$ADB wait-for-device
$ADB shell logcat -c
$ADB shell dmesg -C
$ADB shell log -t WIFI_DBG '==== PHASE: boot wifi_still_off ===='

nohup $ADB logcat -v threadtime -b all \
  WifiService:V WifiScanRequestProxy:V WifiScanningService:V \
  WificondScannerImpl:V WifiNl80211Manager:V WifiNative:V WifiVendorHal:V \
  WifiActiveModeWarden:V WifiController:V WifiClientModeImpl:V \
  HalDevMgr:V RKWifiHAL:V WifiHAL:V wpa_supplicant:V WIFI_DBG:V \
  > /tmp/wifi_2nd_scan.logcat 2>&1 < /dev/null &

nohup $ADB shell dmesg -w > /tmp/wifi_2nd_scan.dmesg 2>&1 < /dev/null &
```

各阶段打标记并快照：

```bash
$ADB shell log -t WIFI_DBG '==== PHASE: first_scan_ok ===='
$ADB shell log -t WIFI_DBG '==== PHASE: disable ===='
$ADB shell log -t WIFI_DBG '==== PHASE: second_scan_fail ===='

$ADB shell getprop wlan.driver.status
$ADB shell getprop init.svc.wpa_supplicant
$ADB shell getprop init.svc.wificond
$ADB shell ip link show wlan0
$ADB shell cat /sys/module/bcmdhd/parameters/firmware_path
$ADB shell cat /sys/module/bcmdhd/parameters/nvram_path
$ADB shell ls -l /vendor/firmware/fw_bcm* /vendor/firmware/fw_bcmdhd.bin /vendor/firmware/nvram*
$ADB shell dumpsys wifi
```

### 2.3 实验 B（手动复位电源脚）

```bash
$ADB shell ls -l /sys/class/rkwifi/
$ADB shell cat /sys/kernel/debug/gpio | grep -iE 'gpiochip|wifi|wlreg|PC4'
# gpio-20 名为 rkwifi_wlan_vbat，关 WiFi 后仍为 out hi

$ADB shell 'echo 0 > /sys/class/rkwifi/wifi_bt_vbat'   # out lo
sleep 1
$ADB shell 'echo 1 > /sys/class/rkwifi/wifi_bt_vbat'   # out hi

$ADB shell svc wifi disable
$ADB shell setprop ctl.restart vendor.wifi_hal_legacy
$ADB shell svc wifi enable
```

### 2.4 编译（修改代码之后）

```bash
cd /home/jx/SourceCode/Android_13
./build.sh -K -A -u
```

产物：`rockdev/Image-rk3588_t/update-rd-box-rk3588-android13-lcd-20260828-115559.img`

---

## 3. 分析出的异常

### 3.1 第二次打开失败点（主因）

关掉再打开时 dmesg：

```
[dhd] PULL WL_REG_ON(-2) HIGH!
[WLAN_RFKILL]: wifi turn on power [GPIO-1-0]
mmc2: tried to HW reset card, got error -110
[dhd] sdio_sw_reset Failed, error = -110
[dhd] wl_android_wifi_on : Failed -110
```

框架侧：

```
Failed to start legacy HAL: UNKNOWN
Failed to create STA iface
WifiClientModeManager: Failed to create ClientInterface. Sit in Idle
```

此时 `wlan0` 为 `DOWN`，`wpa_supplicant` 为 `stopped`，Settings 开关可能仍显示开着，但 `dumpsys wifi` 是 `DisabledState`。

`-110` 是 SDIO 复位超时。第一次开机 probe 时固件已在芯片里，能扫；关掉后再开需要对卡做 `mmc_hw_reset`，卡没响应。

### 3.2 电源脚实际没有被拉（对应实验 A 失败）

- bcmdhd：`WL_REG_ON(-2)`，DTS 的 `bcmdhd_wlan` 没有配 `gpio_wl_reg_on`，打印是空操作。
- rfkill：`rockchip_wifi_power()` 使用 **`WIFI,poweren_gpio`**。板级 DTS 只写了 **`WIFI,vbat_gpio`**（GPIO0_C4），`power_n.io = -1`，因此开关 WiFi **不会翻转 GPIO0_C4**。
- debugfs：关 WiFi 后 `gpio-20 (rkwifi_wlan_vbat) out hi`，模组一直带电。
- 实验 A 多等 10 秒无效：只是软件多停一会儿，硬件没掉电。

### 3.3 实验 B 为何有效

写 `/sys/class/rkwifi/wifi_bt_vbat` 会走 `rfkill_set_wifi_bt_power()`，真正把 GPIO0_C4 拉低再拉高。

仅脉冲、设置里再开时，HAL 仍卡在上次 `-110` 的 Idle，dmesg **没有新的** `wl_android_wifi_on`。  
`ctl.restart vendor.wifi_hal_legacy` 后再 enable，出现：

```
mmc2: queuing unknown CIS tuple ...
mmc_host mmc2: Bus speed ... = 150000000Hz
[dhd] wl_android_wifi_on : Success
```

`wlan0` 变为 UP，设置可扫描。  
结论：需要 **真正掉电复位 SDIO**；HAL 失败后还要能重新起来（电源修好后正常开关不应再卡死）。

### 3.4 固件路径错误（次因，掉电后会暴露）

`init.connectivity.rc` 写入：

- `/vendor/firmware/fw_bcmdhd.bin`
- `/vendor/firmware/nvram.txt`

镜像里实际文件是：

- `/vendor/firmware/fw_bcm4359c0_ag.bin`
- `/vendor/firmware/nvram_ap6398s.txt`

第一次扫描靠开机已加载的固件还能用；按 B 方案掉电复位后必须重载固件，路径必须正确。

---

## 4. 最后修改的文件

| 文件 | 改动 |
|------|------|
| `kernel-5.10/arch/arm64/boot/dts/rockchip/rk3588/rp-wifi-bt-ap6398s-rk3588s.dtsi` | `WIFI,vbat_gpio` 改为 `WIFI,poweren_gpio`（仍为 GPIO0_C4）。同一脚不能两个属性同时申请。不给 `bcmdhd_wlan` 加 `gpio_wl_reg_on`，避免和 rfkill 抢 GPIO。 |
| `kernel-5.10/net/rfkill/rfkill-wlan.c` | `rockchip_wifi_power()` 拉高/拉低 `poweren` 后的 `msleep(100)` 改为 `msleep(200)`。 |
| `device/rockchip/common/init.connectivity.rc` | 两处 firmware/nvram 改为 `fw_bcm4359c0_ag.bin` 与 `nvram_ap6398s.txt`。 |

未改 Settings / WifiService。开关路径应对齐实验 B：关 WiFi 拉低 REG_ON，再开拉高后 `mmc_hw_reset` 能枚举。

---

## 5. 烧录与验证（编译已完成，烧录待板子进 loader）

镜像：

`Android_13/rockdev/Image-rk3588_t/update-rd-box-rk3588-android13-lcd-20260828-115559.img`

烧完冷启动后建议确认：

1. 第一次打开 WiFi 能搜到。
2. 关掉再打开仍能搜到（连续 3 轮）。
3. dmesg：关为 `wifi shut off power [GPIO20-...]`，开为 `wifi turn on power [GPIO20-...]`，**没有** `mmc2 ... error -110`，有 `wl_android_wifi_on : Success`。
4. `cat /sys/module/bcmdhd/parameters/firmware_path` 为 `fw_bcm4359c0_ag.bin`。
