# ext — G3800 双面扩展（Gradle 工程根目录）

**请用 Android Studio 打开本目录：`ext/`**（不要打开上一级解包 APK 根目录 `/mnt/f/print`）。

| 路径 | 说明 |
|------|------|
| `app/` | Android 应用源码 |
| `gradle/` `gradlew*` | Gradle Wrapper |
| `build.gradle.kts` `settings.gradle.kts` | 构建配置 |
| `third_party/canon-sdk/` | 解包 so + jadx 反编译（暂未编入 App） |
| `tools/jadx/` | jadx |
| `scripts/` | SDK 过滤脚本 |
| [BUILD.md](BUILD.md) | 编译说明 |

当前支持在界面切换五种连接方式，并用测试页验证出纸：

| 协议 | 说明 |
|------|------|
| 私有协议 (CLSS/BJNP) | 现有 SNMP+BJNP 发现 + CLSS JPEG |
| IPP Everywhere | NSD + IPP Print-Job（PDF） |
| Raw :9100 | 纯 Java TCP 写 PDF |
| Android 系统打印 | `PrintManager` + 已安装 PrintService |
| Canon Print Service | 同上框架，预检佳能打印插件 |

详见 [BUILD.md](BUILD.md)。
