# 开发环境约束规则 (Skills)

## 1. 操作系统与环境
- 主机：Windows 10
- 已安装 WSL（默认使用 WSL 发行版：Ubuntu-22.04）
- 日常开发在 Windows（PowerShell / CMD）与 WSL 两边都会执行命令
- 在 WSL 中执行 Linux 命令时，注意路径与 Windows 的区别
- 宿主与 WSL 文件系统互通，但 NDK/SDK/临时目录等路径不同，必须按规则使用

## 2. Java 与 JDK 位置
- 项目指定 JDK 版本：21
- Windows JDK 路径：D:/devd/jdk/openlogic-openjdk-21.0.11+10-windows-x64
- WSL JDK 路径：/mnt/d/devd/jdk/openlogic-openjdk-21.0.11+10-linux-x64
- 环境变量 `JAVA_HOME` 必须正确设置，Android Studio 与 Gradle 均使用该 JDK
- 禁止使用系统自带的临时 JDK 或未指定的版本

## 3. 临时目录约束
- **禁止使用系统默认 `/tmp` 或 Windows 临时目录进行编译、缓存或文件操作**
- 所有需要临时存储的场景（如脚本输出、中间构建文件、测试临时文件）必须使用以下指定目录：
  - Windows 临时目录：E:/temp
  - WSL 临时目录：/mnt/e/temp
- 该目录需确保有足够的读写权限，且不应在系统重启后自动清理（除非显式要求）
- Gradle 构建时的缓存、解压等操作若涉及临时目录，需优先使用上述路径

## 4. Android Studio 与 Gradle
- 使用 Android Studio 作为 IDE
- Gradle 版本固定为：9.5
- **严禁切换或升级 Gradle 版本**，所有命令和配置必须兼容此版本
- Gradle 缓存目录（全局）：
  - Windows：E:/gradle
  - WSL：/mnt/e/gradle
- 禁止修改 gradle-wrapper.properties 中的 distributionUrl，如需指定本地发行版，请使用已缓存版本

## 5. Android SDK 位置
- Windows 宿主 SDK 路径：D:/devd/android/sdk-win
- WSL 中 SDK 路径：/mnt/d/devd/android/sdk-linux
- Windows 与 WSL 使用各自独立的 SDK 目录，不可混用
- 环境变量 `ANDROID_HOME` 或 `ANDROID_SDK_ROOT` 应指向当前执行环境对应的路径

## 6. Android NDK 位置
- NDK 版本固定为：30.0.15729638
- Windows NDK 路径：D:/devd/android/sdk-win/ndk/30.0.15729638
- WSL NDK 路径：/mnt/d/devd/android/sdk-linux/ndk/30.0.15729638
- 编译原生代码时必须根据执行上下文选用正确的路径
- 如需在 WSL 中使用 NDK 工具链，确保可执行权限

## 7. 第三方库管理
- 所有第三方库/依赖的本地缓存或手动下载必须放到指定目录：
  - Windows：E:/3rd
  - WSL：/mnt/e/3rd
- 禁止将库下载到项目目录或其他临时目录
- **对于需要编译的第三方源码库（如 C/C++ 库）**：
  - 必须将源码下载或克隆到上述指定目录，并在该目录下完成编译（如使用 CMake、Makefile、NDK 工具链等）
  - **严禁直接使用 `apt`、`yum`、`brew` 等系统包管理器安装预编译库**
  - 若确有例外需求（如系统包管理器安装、改用其他目录），**必须先询问用户再执行**
  - 编译产物（静态库 `.a` / 动态库 `.so`）应从指定目录引用，并正确配置到项目的 CMakeLists.txt 或 Android.mk 中
- 离线模式或手动导入依赖时，需从该目录引用

## 8. 代理配置
- 可用的代理地址：
  - `http_proxy=http://172.19.16.1:4067`
  - `https_proxy=http://172.19.16.1:4067`
- 适用协议：HTTP / HTTPS
- **每次下载前必须先询问用户是否使用代理**，经确认后再设置对应环境变量或工具代理
- 可能需要代理的场景（仍以用户当次确认为准）：
  - Gradle 构建 (gradle.properties 中配置 systemProp)
  - SDK Manager 代理
  - Git 操作
  - 命令行下载工具（如 curl, wget）时设置对应环境变量
- 禁止在代码或配置中硬编码明文用户名/密码（如需要，使用系统环境变量或密钥文件）

## 9. 其他硬性约束
- 禁止修改 Android Studio 自带的 JRE，除非明确指示
- 构建时优先使用命令行 `./gradlew`，必须在项目根目录执行
- 所有生成的文件路径需使用正斜杠或根据系统适配
- 如使用 WSL，注意区分 Windows 可执行文件 (.exe) 与 Linux 二进制

### 执行与编译
- **执行方式**：总是先列出计划，再询问用户是否执行；未经确认不得动手
- **编译**：编译前提示用户，用户可选择手动执行；未经确认不得自行发起编译

### 代码修改
- **禁止整文件重写**，优先局部替换（如精确字符串替换、小范围补丁）
- **禁止“顺手”格式化 / 重命名 / 删注释**，只改与任务直接相关的内容
- 改前必须列出将要修改的文件清单，经用户知悉后再改
- 若项目目录尚未初始化 git，**必须先询问用户**是否初始化或如何处理版本管理，再继续修改
- 改后必须执行 `git diff` 自检，确认变更范围与预期一致
