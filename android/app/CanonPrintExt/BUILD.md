# G3800 手动双面 — 编译说明

研究/自用：Wi‑Fi 上用 Canon 喷墨 **CLSS / IVEC / Port9100** 直连 G3800，做文档手动双面。

> 含从 Canon PRINT 解包的 `libsdk-*.so`，**请勿上架分发**。

## 工程根目录

```
ext/          ← Android Studio / Gradle 根（本文件所在处）
├── app/
├── gradle/
├── third_party/canon-sdk/
├── tools/
└── scripts/
```

上一级 `/mnt/f/print` 是解包 APK，**不是**本工程。

## 当前可编译 / 可运行范围

| 模块 | 状态 |
|------|------|
| Compose UI / PDF 拆页 / 手动双面编排 | 已有 |
| **多协议切换**（私有 / IPP / Raw:9100 / Android 系统 / Canon Print Service） | **已接线** |
| 连接结果 / 具体错误信息区 | **已接线** |
| 测试连接 + 打印测试页 | **已接线** |
| 纸张大小 / 纸张类型设置（持久化，测试页与出纸共用） | **已接线** |
| 喷墨发现（SNMP + BJNP，官方 IJ LAN） | **已接线**（私有协议下「搜索打印机」） |
| 单面 CLSS 出纸（PDF→JPEG→BJNP:8611） | **已接线**（私有协议） |
| 手动双面（两遍单面 + 翻纸） | **已接线**（仅私有协议） |
| Word(.doc/.docx) 云转换 | **已接线**（佳能 CNPS → JPEG → CLSS） |

启动时会 `loadLibrary("sdk-core")`。发现不再只靠 SNMP：与官方喷墨模式一致，并行跑 **SNMP**（native）与 **BJNP UDP:8611**（纯 Java），按 MAC 去重合并。

### 发现路径

```
BroadcastAddress → IjParallelSearch（约 10s）
  ├─ SnmpSearch → libsdk-core StartSNMPSearch
  └─ BjnpSearch → UDP :8611 → BjnpUdp IEEE1284 ID（MFG=Canon / MDL）
→ DiscoveredPrinter（source: SNMP | BJNP | Both）
```

### 单面出纸路径

```
PDF → PdfJpegRenderer → JPEG 页
  → ClssBjnpJpegSession
  → CLSSMakeCommand (JNI) 生成 IVEC XML
  → BjnpSocket(C1631a) TCP :8611
  → StartJob → SetConfiguration(duplex=OFF) → SendData+JPEG* → EndJob
```

### Word 云转换路径（贴近官方 Canon PRINT）

```
选 .doc/.docx → 首次接受云转换说明（prefs: docconvert.accepted.v230）
  → HEAD rs.ciggws.net/select_ps/print → ps_code（3=US / 7=CN / 其它=EU）
  → ATP registerDevice + getAccessToken(scope=oip.prt.AppPrint)
  → POST {prt}/api/mob/1.0/documents/convert
  → PUT gzip 文档 + PrintTicket → 通知 → 轮询 convertjobs
  → GET 每页 data → RC4(EncryptionKey=token 末 16 位) → JPEG
  → ClssBjnpJpegSession.printJpegs
```

PDF **不上云**（本机渲染）。不伪造 `applicationId`；若 ATP 因包名拒绝，UI 会提示。

## Android Studio（推荐）

1. **Open** → 选择 `ext` 文件夹  
2. Sync Gradle（JDK 17、SDK 35）  
3. Build → Build APK(s)  
4. 产物：`app/build/outputs/apk/debug/app-debug.apk`

> 不要混用 WSL 与 Windows 路径编译同一 `build/`（会损坏 Kotlin 增量缓存）。固定一侧 Clean/Rebuild。

## 命令行

```bash
cd ext
cp local.properties.example local.properties   # 编辑 sdk.dir
./gradlew :app:assembleDebug
```

## 真机验证（多协议）

共性：手机与 G3800 同一 Wi‑Fi；允许局域网 / 位置权限。先选好 **纸张大小 / 纸张类型**（默认 A4·普通纸，会记住），再按每种协议：**搜索 → 看「连接结果 / 错误」→ 测试连接 → 打印测试页**。测试页 PDF 尺寸与 CLSS/IPP/系统 MediaSize 均跟随该设置。失败时应看到具体错误（端口拒绝、IPP status、插件未安装等），而非笼统「失败」。

1. **私有协议**：搜索（SNMP+BJNP）→ 测试连接（完整 BJNP 会话：UDP 408B 握手 + TCP:8611）→ 测试页出纸；手动双面仍可用  
   - 诊断：`adb logcat -s G3800Bjnp`（open / openRetry / probe / session）  
   - 注意：会话打开包必须是 **408** 字节（官方 `BjnpSocket`）；误发 24 字节会导致 UDP 无应答 / 连不上
2. **IPP**：NSD `_ipp._tcp`；若为空会回退用 Canon 发现的 IP 探 `ipp://IP:631/ipp/print`  
3. **Raw:9100**：NSD / Canon IP 探 TCP:9100 → 测试页写 PDF 流（固件不接受 PDF 时错误区会显示发送异常或无反应说明）  
4. **Android 系统打印**：列出已安装 PrintService → 测试页弹出系统对话框 → 选打印机  
5. **Canon Print Service**：预检包名 `jp.co.canon.android.printservice.plugin`（未安装/未启用有明确文案）→ 测试页在对话框中选该服务下的打印机  

## 真机验证（私有协议发现细节）

1. 选「私有协议」→「搜索打印机」→ 约 10s  
2. 列表应显示 IP、MAC、来源（SNMP / BJNP / SNMP+BJNP）  
3. 对照：仅 SNMP、仅 BJNP、两路命中去重为一条  
4. 「单面 CLSS 出纸」仍应可用

## 真机验证（Word .doc / .docx）

1. 同网 G3800；手机可访问外网（ATP + CNPS）  
2. 「选择 PDF/Word」→ 选一份 `.docx`；首次弹出云转换说明 → 接受  
3. 状态应出现「云端转换…」→ 成功后显示 JPEG 页数  
4. 「单面 CLSS 出纸」→ 应对应页出纸  
5. 再选一份旧格式 `.doc` 重复步骤 2–4  
6. 「开始手动双面打印」→ 正面出纸 → 按提示翻面 → 背面出纸  
7. 若 ATP 注册失败且文案含包名拒绝：属预期风险，勿改 `applicationId` 伪装官方包  

### Word 转换失败排错（logcat）

过滤 tag：`G3800CloudConvert`（或 `adb logcat -s G3800CloudConvert`）。  
会输出阶段（`atp` / `select_ps` / `createJob` / `putDocument` / `poll` / `download` 等）、HTTP 状态与响应片段、异常栈与 cause 链。界面「连接结果 / 错误」区也会显示同一详细信息。

## 相关源码

- `transport/`：`PrintProtocol` / `PrinterBackend` / `CanonClssBackend` / `IppBackend` / `Raw9100Backend` / `AndroidPrintBackend` / `IppClient` / `NsdServiceDiscovery`
- `print/PrintPaperSettings.kt`：纸张大小/类型与各协议映射；`PaperSettingsStore` 持久化
- `print/TestPageGenerator.kt`：按所选尺寸生成测试页 PDF
- `ui/DuplexPrintScreen.kt`：协议切换、连接结果、测试连接 / 测试页
- `cloud/CanonAtpAuth.kt` / `CanonCloudDocConverter.kt` / `PrintTicketFactory.kt` / `Rc4Utility.kt`
- `canon/CanonSdkBridge.kt`（`normalizeToPrintable` / `printSimplexDocument`）
- `duplex/DuplexPrintController.kt`（`runJpegs` / `PageOrder` 奇偶拆正反）
- 移植：`jp/co/canon/oip/android/opal/mobileatp/**`（HTTP 经精简 `b/C1204a`）
