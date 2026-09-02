# WSL2 内网固定 IP 脚本

将 WSL2 的内网地址固定为 `172.19.16.0/20`。

WSL2 NAT 子网由服务启动时自动分配，`.wslconfig` 无法指定网关 IP。本方案在每次启动后把宿主 IP、访客 IP、IPv4 转发和 WinNAT 一起纠正。

## 目标拓扑

| 位置 | 固定后 |
| ---- | ------ |
| 宿主机 `vEthernet (WSL)` | **172.19.16.1 / 255.255.240.0** |
| WSL eth0 | **172.19.16.2 / 20** |
| 默认网关 / DNS | 172.19.16.1 |
| WinNAT 名称 / 内网前缀 | `WslFixNat17219` / `172.19.16.0/20` |
| 发行版 | `Ubuntu-2204`（`wsl-fix-net.cmd` 里的 `WSL_DISTRO`） |

## 文件

| 文件 | 作用 |
| ---- | ---- |
| `wsl-fix-net.cmd` | 入口。需管理员（会自动提权）。可选参数 `nopause` 跳过结束时的暂停，供计划任务使用。GBK 编码。 |
| `wsl-host-net.ps1` | 宿主：`netsh` 静态 IP、在 WSL 网卡和默认路由网卡上开 IPv4 转发、按前缀创建/复用 NetNat、入站防火墙规则 `WSL-FixNet-Allow-vEthernet`。UTF-8。 |
| `wsl-guest-fix.sh` | 访客：eth0 静态地址、默认路由、DNS；必要时写 `/etc/wsl.conf` 的 `generateResolvConf = false`。UTF-8 / LF。 |
| `setup-task.cmd` | 注册当前用户登录任务 `WslFixNet`（不要用 SYSTEM）。GBK。 |
| `uninstall-task.cmd` | 只删除计划任务 `WslFixNet`；还原网络需按它打印的命令手工执行。GBK。 |

## 使用

1. 管理员运行一次 `wsl-fix-net.cmd`，确认网关和外网都通。
2. 管理员运行 `setup-task.cmd`，注册登录后自动纠正。

`wsl-fix-net.cmd` 实际步骤：

1. 无引号启动 `wsl -d %WSL_DISTRO% -u root --`，确认发行版能起来。
2. 调用 `wsl-host-net.ps1` 设宿主 IP / 转发 / NAT。
3. 用 `wslpath -a` 把 `wsl-guest-fix.sh` 的 Windows 路径转成 `/mnt/...`。
4. 在发行版内 `bash` 跑访客脚本。
5. 打印 `netsh` / `ip`，并 `ping` 网关和 `8.8.8.8`。

改 IP 后脚本会 `ping 127.0.0.1 -n 3` 短暂等待，避免网关 ICMP 还没就绪。

## 计划任务（与脚本一致）

- 任务名：`WslFixNet`
- 触发：当前用户 **ONLOGON**，延迟 **20 秒**
- `schtasks /DELAY` 格式必须是 **`mmmm:ss`（分钟:秒）**，本脚本为 `/DELAY 0000:20`。写成 `0000:00:20`（时:分:秒）会报“DELAY 值无效”，任务创建失败。
- 动作：`wsl-fix-net.cmd nopause`，运行级别 Highest
- 不用 `/RU SYSTEM`：发行版是用户级的，SYSTEM 会 `DISTRO_NOT_FOUND`；也不用 ONSTART，开机过早时 `vEthernet (WSL)` 可能还不存在

已有同名任务时，`setup-task.cmd` 会先删除再重建。

## 本机踩过的坑（脚本已按此写）

1. `wsl -d "Ubuntu-2204"` **带引号** 在 Win10 + WSL 2.7 上会 `WSL_E_DISTRO_NOT_FOUND`。发行版名不加引号，参数用 `--` 分隔。
2. Windows 路径不能直接交给 WSL `bash`（反斜杠会被吃掉）。必须 `wslpath -a`。
3. 只改 `vEthernet (WSL)` 的 IP 会出站断网（网关能 ping，外网不通）。必须同时：
   - 在 `vEthernet (WSL)` 和默认路由网卡上开启 IPv4 Forwarding
   - 为 `172.19.16.0/20` 创建 NetNat（本机原先可能还有无关的 `WSLNat` / `192.168.50.0/24`，不要指望它覆盖新网段）
4. `.cmd` 中文用 **GBK**；`.ps1` / `.sh` / `README.md` 用 UTF-8。

## 验证

```cmd
netsh interface ipv4 show config name="vEthernet (WSL)"
wsl -d Ubuntu-2204 -- ip -4 addr show dev eth0
wsl -d Ubuntu-2204 -- ping -c 2 -W 2 172.19.16.1
wsl -d Ubuntu-2204 -- ping -c 2 -W 2 8.8.8.8
```

预期：宿主 `172.19.16.1/20`，访客 `172.19.16.2/20`，两条 ping 0% 丢包。

## 改参数

三处要一起改，否则宿主/访客会对不齐：

1. `wsl-fix-net.cmd` 顶部：`IFACE`、`HOST_IP`、`MASK`、`GUEST_IP`、`GUEST_MASK`、`NAT_PREFIX`、`NAT_NAME`、`WSL_DISTRO`
2. `wsl-guest-fix.sh`：`GATEWAY`、`IP`、`MASK`、`IFACE`（访客侧是硬编码，不会读 cmd 变量）
3. 若改了 `NAT_NAME`，同步改 `uninstall-task.cmd` 提示里的 `WslFixNat17219`

发行版名用 `wsl --list --verbose` 查看；不要给 `-d` 加引号。

## 卸载

运行 `uninstall-task.cmd` 只会删掉任务 `WslFixNet`。要恢复 WSL 自动分配，再执行它打印的命令（管理员 PowerShell / cmd）：

```powershell
Get-NetNat -Name WslFixNat17219 -ErrorAction SilentlyContinue | Remove-NetNat -Confirm:$false
```

```cmd
netsh interface ipv4 set address name="vEthernet (WSL)" dhcp
wsl --shutdown
```

然后重新打开 WSL。入站规则 `WSL-FixNet-Allow-vEthernet` 如需一并去掉，可在“高级安全 Windows 防火墙”里删除，或：

```powershell
Get-NetFirewallRule -DisplayName "WSL-FixNet-Allow-vEthernet" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
```