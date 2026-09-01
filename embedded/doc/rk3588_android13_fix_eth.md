# RK3588 网线无法分配 IP / 达不到千兆：排查与修复

产品：`rk3588_t` / 板卡 `rd-box-rk3588`  
MAC/PHY：`rk_gmac-dwmac`（GMAC1，`ethernet@fe1c0000`）+ RTL8211F（RGMII，`phy-mode = rgmii-rxid`）  
现象：

1. 插上网线后链路经常落在 **10 Mbps**（PHY 本身是千兆）。
2. 烧完一版 `boot.img` 后链路一度到 **1000 Mbps**，但 **eth0 分不到 IP**，远端 adb 也跟着断。

---

## 1. 排查过程

### 1.1 环境

- 源码与编译：本机 `192.168.66.130`（`/home/jx/SourceCode/Android_13`）。
- Android 板挂在远端 `192.168.66.112`，登录：`ssh -p 2201 shenda@192.168.66.112`。
- 必须用 `/mnt/d/devd/platform-tools/adb.exe`（Linux `/usr/bin/adb` 看不到这台 USB 设备）。
- 约定：只读可连着做；改线 / 改 PHY 寄存器 / 重启 / 烧录前再问。

### 1.2 分层思路

不先改 delay。先分清是线/交换机、PHY 自协商，还是 RGMII 时序，再看 Android 是否给 eth0 跑 DHCP：

线 / hub / 对端 → PHY 自协商寄存器 → RGMII `tx_delay` → 内核 Link Up → EthernetTracker / dhcpcd → IPv4

对照实验：同一 hub、同一根网线，接对照板 `tl3588_evm`。

| 实验 | 做法 | 结果 |
|------|------|------|
| 对照板 | 同 hub 同网线接 `tl3588_evm` | **1000 Mbps full**，排除线/hub |
| 本板只读 | speed / dmesg / DTS | 长期 **10 Mbps**；DT 为 `rgmii-rxid`、`tx_delay=0x44`；从未出现过 1000M link |
| PHY 寄存器 | 读 RTL8211F 自协商 | 本端 CTRL1000=`0x200`（已宣告 1000Full）；对端 LPA 当时只见 10M |
| 只宣告 1000 | 关掉 10/100 能力 | 链路 **down**；对端 GBSR 仍有 1000Full |
| 强制 1000Full+Master | 写 PHY | 仍 **down** |
| DTS `tx_delay` 0x44→0x45 | 只编 DTB、重打 boot、只烧 boot | 板上 `/proc/device-tree` **仍是 0x44**；重启后链路变成 **1000M**，随后 **分不到 IP** |

### 1.3 关键踩坑：改了 DTB 内核没吃到

RK3588 U-Boot 用的是 boot.img 里的 **`second`（resource.img → `rk-kernel.dtb`）**。  
只替换 Android header v2 的 `--dtb` 槽，运行时 `tx_delay` 仍是旧值。

所以后来千兆起来，**不是** `0x45` 的效果，是旧 delay `0x44` 下重启后协商成功。`tx_delay` 走 RGMII，本来也不决定 1000Base-T 自协商（自协商走 RJ45 四对线）。

---

## 2. 运行过的命令

`$ADB` = `/mnt/d/devd/platform-tools/adb.exe`，经 `ssh -p 2201 shenda@192.168.66.112`。

### 2.1 链路快照

```bash
$ADB shell cat /sys/class/net/eth0/speed
$ADB shell cat /sys/class/net/eth0/duplex
$ADB shell cat /sys/class/net/eth0/operstate
$ADB shell cat /sys/class/net/eth0/carrier
$ADB shell ip -br addr
$ADB shell ip -br link
$ADB shell ifconfig eth0
$ADB shell dmesg | grep -iE 'gmac|eth0|stmmac|Link is|RTL8211|TX delay'
$ADB shell cat /proc/device-tree/ethernet@fe1c0000/phy-mode
$ADB shell cat /proc/device-tree/ethernet@fe1c0000/tx_delay | od -An -tx1
```

### 2.2 对照板

```bash
$ADB devices -l
# 对照板 serial：6de2bf69a1dbbc4d（tl3588_evm）
$ADB -s 6de2bf69a1dbbc4d shell cat /sys/class/net/eth0/speed   # 1000
```

### 2.3 IP / DHCP（分配不到 IP 时）

```bash
$ADB shell ip -br addr
$ADB shell getprop | grep -iE 'eth|dhcp|net.dns|gateway'
$ADB shell dumpsys ethernet
$ADB shell getprop init.svc.dhcpcd_eth0
$ADB shell ls /sys/class/net
```

Android 13 上 eth0 的 IPv4 通常由 **EthernetTracker / NetworkStack DHCP** 分配，不是 Settings 里的 WiFi。`init.rockchip.rc` 里有 `dhcpcd_eth0`，默认 **disabled**。`init.rk3588.rc` 有 `start_ethernet`（打开以太网设置页），也是 disabled。

链路 Up 但没 IPv4 时，优先看：

- `operstate` / `carrier` 是否真 Up
- `dumpsys ethernet` 是否在跑 DHCP、是否失败
- 链路是否刚从 10/100 抖到 1000（DHCP 可能没在新链路上重试）

### 2.4 只编 DTB 并重打 boot

```bash
cd /home/jx/SourceCode/Android_13
make -C kernel-5.10 ARCH=arm64 \
    O=out/target/product/rk3588_t/obj/KERNEL_OBJ dtbs

# 解包后只换 --dtb 槽（这次不够：内核实际吃 resource.img）
unpack_bootimg --boot_img out/target/product/rk3588_t/boot.img \
    --out $WORK/unpacked --format=mkbootimg
```

回滚副本：`out/target/product/rk3588_t/boot.img.pre-txdelay045`。

---

## 3. 分析出的异常

### 3.1 长期只有 10M（主因，硬件/训练）

- PHY 是千兆 RTL8211F，本端已宣告 1000Full。
- 对照板同线同 hub 能 1000M，排除线/交换机。
- 只宣告或强制 1000 时本板链路起不来：对端有千兆能力，本板 **1000Base-T 训练失败**。
- 更像变压器 / MDI2-MDI3 四对走线 / PHY 模拟，而不是 DTS delay 写错。

### 3.2 千兆偶发起来之后分不到 IP

烧 boot 重启后：

| 检查项 | 结果 |
|--------|------|
| speed / duplex / operstate | **1000 / full / up** |
| dmesg | 先 `100Mbps` 掉线，再 `Link is Up - 1Gbps/Full` |
| `/proc/device-tree/.../tx_delay` | 仍是 `00 00 00 44` |
| IPv4 | **没有**（原先 10M 时是 `192.168.0.102/24`） |

链路从 100M 掉到再 1000M，EthernetTracker 若没在新 carrier 上重跑 DHCP，就会出现「网线亮了、没 IP」。这和 delay 改没改无关。

### 3.3 为什么 0x45 没生效

只换了 header v2 `--dtb`。U-Boot 加载的是 `second` 里的 `rk-kernel.dtb`。  
要用 `resource_tool` 换 `rk-kernel.dtb` 再打进 boot.img，或走完整 `./build.sh -CK`（会重打 resource）。

---

## 4. 最后修改的文件

| 文件 | 改动 | 运行时是否生效 |
|------|------|----------------|
| `kernel-5.10/arch/arm64/boot/dts/rockchip/rk3588/rp-eth-gmac1.dtsi` | `tx_delay` `0x44` → `0x45`（`rx_delay` 仍注释） | **否**。板上仍是 0x44。源码保留该值作对照。 |

未改驱动、未改 `dt-overlay.in`、未改 `dhcpcd` / Ethernet overlay。

以太网 IP 要恢复时（不改 DTS）：

1. 确认 `eth0` carrier=1、speed 合理。
2. `dumpsys ethernet` 看 DHCP。
3. 必要时在设置里打开以太网，或 `svc` / Ethernet 设置页触发一次。
4. 不要用 `ifconfig eth0 <static>` 当长期方案，除非产品明确要静态 IP。

---

## 5. 验证要点

1. `cat /sys/class/net/eth0/speed` 与 dmesg `Link is Up` 一致。
2. `ip -br addr` 里 eth0 有 IPv4；`dumpsys ethernet` 不是一直 `DISCONNECTED` / DHCP 失败。
3. 若改了 DTS：必须同时确认  
   `cat /proc/device-tree/ethernet@fe1c0000/tx_delay | od -An -tx1`  
   以及 dmesg `TX delay(0x..)`，**不要只看源码**。
4. 改 DTB 只烧 boot 时，检查 `second`/resource 里的 `rk-kernel.dtb`，不要只换 `--dtb`。
