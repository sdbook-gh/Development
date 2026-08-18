#!/usr/bin/env bash
# =============================================================================
# build.sh
#
# 功能:
#   - 可指定 Gradle 版本 / 分发下载地址 (多镜像 fallback: 腾讯->阿里->华为->官方)
#   - 可指定 Gradle 缓存位置 (默认当前目录下 gradle_cache/), 完全隔离系统 ~/.gradle
#   - 可指定 Android SDK / NDK 位置, 自动适配 WSL(Linux) 与 Windows(Git Bash)
#   - 可指定下载代理 (用于分发下载 + Gradle 依赖下载)
#   - clean / clean-all 清理编译产物, 最小空间占用
#
# 用法: ./build.sh [命令] [选项...] [-- 额外gradle参数...]
#   命令: build(默认) | release | clean | clean-all | help
#   选项见 --help
# =============================================================================

set -uo pipefail

# ----------------------------- 初始化变量 -------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "无法进入项目目录: $SCRIPT_DIR"; exit 1; }

COMMAND=""
GRADLE_ARGS=()
NO_DAEMON=0
SDK=""
NDK_DIR=""
TMP_DIR=""
PROXY_INIT_FILE=""
OFFLINE=0
COPY_APK=0

# 颜色输出
log()  { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[err ]\033[0m %s\n' "$*" >&2; }

# ----------------------------- 用法说明 ---------------------------------------
usage() {
  cat <<'EOF'
用法: ./build.sh [命令] [选项...] [-- 额外gradle参数...]

命令:
  build            构建 Debug APK (assembleDebug, 默认)
  release          构建 Release APK (assembleRelease)
  clean            清理编译产物 (保留 gradle_cache 与源码/配置)
  clean-all        深度清理: 编译产物 + 整个 gradle_cache/ (最小空间占用)
  help             显示本帮助

选项 (均可通过同名环境变量覆盖, 命令行参数优先):
  -v, --gradle-version <ver>   Gradle 版本, 默认 9.5.0
      --gradle-dist-url <url>  Gradle 分发下载地址 (指定则仅用该地址)
                               默认多镜像 fallback: 腾讯->阿里->华为->官方
  -c, --gradle-cache <dir>     Gradle 缓存目录, 默认 <项目>/gradle_cache
      --sdk <path>             直接指定当前环境 Android SDK (优先级最高)
      --sdk-linux <path>       WSL/Linux 下 SDK, 默认 /mnt/d/devd/android/sdk-linux
      --sdk-win <path>         Windows(Git Bash) 下 SDK, 默认 D:/devd/android/sdk-win
      --ndk-version <ver>      NDK 版本, 默认 30.0.15729638
      --ndk-dir <path>         直接指定 NDK 目录 (优先级最高)
  -p, --proxy <url>            下载代理, 如 http://172.19.16.1:4067
  -j, --java-home <path>       JDK 路径, 默认按环境探测 (JDK 21)
      --tmp <dir>              临时目录, 默认 WSL:/mnt/e/temp  Win:E:/temp
      --no-daemon              不使用 Gradle daemon
      --offline                仅使用 gradle_cache, 不下载任何依赖
  -o, --copy-apk              构建完成后将 APK 复制到当前工作目录

环境变量: GRADLE_VERSION, GRADLE_DIST_URL, GRADLE_CACHE, SDK, SDK_LINUX, SDK_WIN,
          NDK_VERSION, NDK_DIR, PROXY, JAVA_HOME, TMP_DIR, OFFLINE

示例:
  ./build.sh                                   # Debug 构建
  ./build.sh release                           # Release 构建
  ./build.sh --offline build                   # 离线构建 (仅用 gradle_cache)
  ./build.sh -v 9.5.0 -c ./gradle_cache        # 指定版本与缓存
  ./build.sh --proxy http://172.19.16.1:4067   # 使用代理
  ./build.sh clean                             # 清理产物
  ./build.sh clean-all                         # 清理产物 + 缓存
  ./build.sh lint -- --info                    # 透传 gradle 参数
EOF
}

# ----------------------------- 参数解析 ---------------------------------------
parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help|help)   COMMAND="help"; shift;;
      build|assembleDebug) COMMAND="build"; shift;;
      release|assembleRelease) COMMAND="assembleRelease"; shift;;
      stop)     COMMAND="stop"; shift;;
      clean)            COMMAND="clean"; shift;;
      clean-all|cleanall) COMMAND="clean-all"; shift;;
      -v|--gradle-version) GRADLE_VERSION="$2"; shift 2;;
      --gradle-dist-url)   GRADLE_DIST_URL="$2"; shift 2;;
      -c|--gradle-cache)   GRADLE_CACHE="$2"; shift 2;;
      --sdk)               SDK="$2"; shift 2;;
      --sdk-linux)         SDK_LINUX="$2"; shift 2;;
      --sdk-win)           SDK_WIN="$2"; shift 2;;
      --ndk-version)       NDK_VERSION="$2"; shift 2;;
      --ndk-dir)           NDK_DIR="$2"; shift 2;;
      -p|--proxy)          PROXY="$2"; shift 2;;
      -j|--java-home)      JAVA_HOME="$2"; shift 2;;
      --tmp)               TMP_DIR="$2"; shift 2;;
      --no-daemon)         NO_DAEMON=1; shift;;
      --offline)           OFFLINE=1; shift;;
      -o|--copy-apk)       COPY_APK=1; shift;;
      --) shift; GRADLE_ARGS+=("$@"); break;;
      *) GRADLE_ARGS+=("$1"); shift;;
    esac
  done
  [[ -z "$COMMAND" ]] && COMMAND="build"
}

# ----------------------------- 环境探测 ---------------------------------------
detect_platform() {
  local sys; sys="$(uname -s)"
  case "$sys" in
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows";;
    *) PLATFORM="linux";;   # 含 WSL 与原生 Linux
  esac
}

# ----------------------------- 路径与默认值解析 -------------------------------
resolve_defaults() {
  : "${GRADLE_VERSION:=9.5.0}"
  : "${SDK_LINUX:=/mnt/d/devd/android/sdk-linux}"
  : "${SDK_WIN:=D:/devd/android/sdk-win}"
  : "${NDK_VERSION:=30.0.15729638}"

  # JAVA_HOME 默认按平台
  if [[ -z "${JAVA_HOME:-}" ]]; then
    case "$PLATFORM" in
      linux)   JAVA_HOME="/mnt/d/devd/jdk/openlogic-openjdk-21.0.11+10-linux-x64";;
      windows) JAVA_HOME="D:/devd/jdk/openlogic-openjdk-21.0.11+10-windows-x64";;
    esac
  fi

  # SDK 默认按平台 (除非 --sdk 强制指定)
  if [[ -z "${SDK:-}" ]]; then
    case "$PLATFORM" in
      linux)   SDK="$SDK_LINUX";;
      windows) SDK="$SDK_WIN";;
    esac
  fi

  # NDK 默认由 SDK 推导
  if [[ -z "${NDK_DIR:-}" ]]; then
    NDK_DIR="$SDK/ndk/$NDK_VERSION"
  fi

  # 临时目录
  if [[ -z "${TMP_DIR:-}" ]]; then
    case "$PLATFORM" in
      linux)   TMP_DIR="/mnt/e/temp";;
      windows) TMP_DIR="E:/temp";;
    esac
  fi

  # Gradle 缓存默认当前目录下 gradle_cache
  if [[ -z "${GRADLE_CACHE:-}" ]]; then
    GRADLE_CACHE="$SCRIPT_DIR/gradle_cache"
  fi

  # GRADLE_DIST_URL 默认留空 -> ensure_gradle 内部使用多镜像 fallback
}

# ----------------------------- JDK 校验 ---------------------------------------
validate_java() {
  if [[ ! -d "$JAVA_HOME" ]]; then
    err "JAVA_HOME 不存在: $JAVA_HOME"
    exit 1
  fi
  local javacmd="$JAVA_HOME/bin/java"
  [[ "$PLATFORM" == "windows" ]] && javacmd="$JAVA_HOME/bin/java.exe"
  if ! "$javacmd" -version >/dev/null 2>&1; then
    err "JDK 不可用: $javacmd"
    exit 1
  fi
  log "JAVA_HOME = $JAVA_HOME"
}

# ----------------------------- 文件下载 ---------------------------------------
download_file() {
  local url="$1" dest="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --connect-timeout 30 --retry 3 --continue-at - -# -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --timeout=30 --tries=3 -c -O "$dest" "$url"
  else
    err "未找到 curl 或 wget, 无法下载"
    return 1
  fi
}

# ----------------------------- Gradle 分发管理 --------------------------------
ensure_gradle() {
  local dist_dir="$GRADLE_CACHE/gradle-$GRADLE_VERSION"
  local gradle_bin="$dist_dir/bin/gradle"

  if [[ -x "$gradle_bin" ]]; then
    log "复用已存在的 Gradle 分发: $dist_dir"
    return 0
  fi

  # 分发下载地址列表: --gradle-dist-url 指定则仅用该地址, 否则多镜像 fallback
  local -a urls=()
  if [[ -n "${GRADLE_DIST_URL:-}" ]]; then
    urls+=("$GRADLE_DIST_URL")
  else
    local v="$GRADLE_VERSION"
    urls+=("https://mirrors.cloud.tencent.com/gradle/gradle-${v}-bin.zip")
    urls+=("https://mirrors.aliyun.com/gradle/gradle-${v}-bin.zip")
    urls+=("https://mirrors.huaweicloud.com/gradle/gradle-${v}-bin.zip")
    urls+=("https://services.gradle.org/distributions/gradle-${v}-bin.zip")
  fi

  log "未找到 Gradle $GRADLE_VERSION, 开始下载..."
  log "  缓存: $GRADLE_CACHE"

  mkdir -p "$GRADLE_CACHE" "$TMP_DIR"
  local tmp_zip="$TMP_DIR/gradle-$GRADLE_VERSION-bin.zip"
  local tmp_extract="$TMP_DIR/gradle-extract-$$"
  local ok=0 url
  for url in "${urls[@]}"; do
    log "  尝试镜像: $url"
    rm -f "$tmp_zip"
    if download_file "$url" "$tmp_zip"; then
      ok=1
      log "  下载成功"
      break
    fi
    warn "  该镜像下载失败, 尝试下一个..."
  done
  if [[ "$ok" != 1 ]]; then
    err "所有镜像下载失败, 请检查网络或使用 --proxy / --gradle-dist-url"
    rm -rf "$tmp_extract" "$tmp_zip"
    exit 1
  fi

  if ! command -v unzip >/dev/null 2>&1; then
    err "未找到 unzip, 请安装后重试"
    rm -rf "$tmp_extract" "$tmp_zip"
    exit 1
  fi

  log "解压中..."
  rm -rf "$tmp_extract"; mkdir -p "$tmp_extract"
  unzip -q "$tmp_zip" -d "$tmp_extract"

  # zip 内顶层目录通常为 gradle-<ver>, 取第一个目录并重命名为标准名
  local extracted
  extracted="$(find "$tmp_extract" -maxdepth 1 -mindepth 1 -type d | head -1)"
  if [[ -z "$extracted" ]]; then
    err "解压后未找到目录, 请检查分发 zip"
    rm -rf "$tmp_extract" "$tmp_zip"
    exit 1
  fi

  rm -rf "$dist_dir"
  mkdir -p "$GRADLE_CACHE"
  mv "$extracted" "$dist_dir"
  chmod +x "$dist_dir/bin/gradle" 2>/dev/null || true
  rm -rf "$tmp_extract" "$tmp_zip"

  if [[ ! -x "$gradle_bin" ]]; then
    err "Gradle 安装异常: $gradle_bin 不可执行"
    exit 1
  fi
  log "Gradle $GRADLE_VERSION 就绪: $dist_dir"
}

# ----------------------------- SDK / NDK 配置 ---------------------------------
write_local_properties() {
  local sdk="$1"
  local line="sdk.dir=$sdk"
  # 仅当与现有值不同才写入, 避免无谓改动
  if [[ -f local.properties ]] && grep -Fqx "$line" local.properties 2>/dev/null; then
    log "local.properties 已为最新, 跳过写入"
    return 0
  fi
  printf '%s\n' "$line" > local.properties
  log "已写入 local.properties: $line"
}

setup_sdk_ndk() {
  export ANDROID_HOME="$SDK"
  export ANDROID_SDK_ROOT="$SDK"
  export ANDROID_NDK_HOME="$NDK_DIR"
  export ANDROID_NDK_ROOT="$NDK_DIR"

  if [[ ! -d "$SDK" ]]; then
    err "Android SDK 不存在: $SDK"
    exit 1
  fi
  if [[ ! -d "$NDK_DIR" ]]; then
    warn "NDK 目录不存在: $NDK_DIR (本项目若不编译原生代码可忽略)"
  fi
  log "Android SDK = $SDK"
  log "Android NDK = $NDK_DIR"
  write_local_properties "$SDK"
}

# ----------------------------- 代理配置 ---------------------------------------
setup_proxy() {
  if [[ -z "${PROXY:-}" ]]; then
    return 0
  fi
  log "使用代理: $PROXY"
  export http_proxy="$PROXY" https_proxy="$PROXY"
  export HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
  export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"

  # 解析 host / port
  local stripped="${PROXY#http://}"
  stripped="${stripped#https://}"
  stripped="${stripped%/}"
  local host="${stripped%%:*}"
  local port=""
  if [[ "$stripped" == *:* ]]; then port="${stripped##*:}"; fi

  mkdir -p "$GRADLE_CACHE"
  PROXY_INIT_FILE="$GRADLE_CACHE/init-proxy.gradle"
  {
    echo "// 自动生成 by build.sh - Gradle JVM 代理设置"
    echo "System.setProperty(\"http.proxyHost\", \"$host\")"
    if [[ -n "$port" ]]; then
      echo "System.setProperty(\"http.proxyPort\", \"$port\")"
    fi
    echo "System.setProperty(\"https.proxyHost\", \"$host\")"
    if [[ -n "$port" ]]; then
      echo "System.setProperty(\"https.proxyPort\", \"$port\")"
    fi
    echo "System.setProperty(\"http.nonProxyHosts\", \"localhost|127.0.0.1\")"
  } > "$PROXY_INIT_FILE"
  log "已生成 Gradle 代理 init 脚本: $PROXY_INIT_FILE"
}

# ----------------------------- 项目优化 (中国大陆加速) -------------------------
optimize_repos() {
  local file="settings.gradle.kts"
  [[ -f "$file" ]] || { file="settings.gradle"; [[ -f "$file" ]] || return 0; }

  grep -q "maven.aliyun.com" "$file" 2>/dev/null && return 0

  [[ -f "$file.bak" ]] || { cp "$file" "$file.bak"; log "已备份: $file -> $file.bak"; }

  mkdir -p "$TMP_DIR"
  local tmp="$TMP_DIR/opt-settings-$$"
  local found_plugin=0 found_dep=0
  local marker='maven { url = uri("https://mirrors.cloud.tencent.com/nexus/repository/maven-public/") }'

  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '%s\n' "$line" >> "$tmp"
    if [[ "$line" == *"$marker"* ]]; then
      if [[ $found_plugin -eq 0 ]]; then
        printf '%s\n' '        maven { url = uri("https://maven.aliyun.com/repository/google") }' >> "$tmp"
        printf '%s\n' '        maven { url = uri("https://maven.aliyun.com/repository/central") }' >> "$tmp"
        printf '%s\n' '        maven { url = uri("https://maven.aliyun.com/repository/gradle-plugin") }' >> "$tmp"
        printf '%s\n' '        maven { url = uri("https://mirrors.huaweicloud.com/repository/maven/") }' >> "$tmp"
        found_plugin=1
      elif [[ $found_dep -eq 0 ]]; then
        printf '%s\n' '        maven { url = uri("https://maven.aliyun.com/repository/google") }' >> "$tmp"
        printf '%s\n' '        maven { url = uri("https://maven.aliyun.com/repository/central") }' >> "$tmp"
        printf '%s\n' '        maven { url = uri("https://mirrors.huaweicloud.com/repository/maven/") }' >> "$tmp"
        found_dep=1
      fi
    fi
  done < "$file"

  mv "$tmp" "$file"
  log "已优化仓库镜像: $file (阿里云 + 华为国内镜像)"
}

optimize_gradle_properties() {
  local file="gradle.properties"
  [[ -f "$file" ]] || return 0

  grep -q "kotlin.incremental" "$file" 2>/dev/null && return 0

  [[ -f "$file.bak" ]] || { cp "$file" "$file.bak"; log "已备份: $file -> $file.bak"; }

  local changed=0

  if ! grep -q "UseG1GC" "$file" 2>/dev/null; then
    sed -i 's/org.gradle.jvmargs=-Xmx[^ ]*/& -XX:+UseG1GC -XX:MaxGCPauseMillis=200/' "$file"
    changed=1
  fi

  if ! grep -q "kotlin.incremental" "$file" 2>/dev/null; then
    # 确保文件以换行结尾，避免追加内容粘连到上一行（原文件末行无换行时）
    [[ -s "$file" && -n "$(tail -c1 "$file")" ]] && echo >> "$file"
    echo "kotlin.incremental=true" >> "$file"
    changed=1
  fi

  if ! grep -q "kotlin.daemon.jvmargs" "$file" 2>/dev/null; then
    echo "kotlin.daemon.jvmargs=-Xmx2048m" >> "$file"
    changed=1
  fi

  [[ $changed -eq 1 ]] && log "已优化 gradle.properties (GC + Kotlin daemon)"
}

optimize_project() {
  log "分析项目配置..."
  optimize_repos
  optimize_gradle_properties
}

# ----------------------------- 缓存隔离与构建优化 -----------------------------
write_gradle_properties() {
  # 读取项目 gradle.properties 中的关键属性, 同步到缓存目录
  mkdir -p "$GRADLE_CACHE"
  local proj_jvmargs="org.gradle.jvmargs=-Xmx4096m -Dfile.encoding=UTF-8"
  local proj_parallel="org.gradle.parallel=true"
  local proj_caching="org.gradle.caching=true"
  if [[ -f "gradle.properties" ]]; then
    local v; v=$(grep "^org.gradle.jvmargs=" gradle.properties 2>/dev/null | head -1); [[ -n "$v" ]] && proj_jvmargs="$v"
    v=$(grep "^org.gradle.parallel=" gradle.properties 2>/dev/null | head -1); [[ -n "$v" ]] && proj_parallel="$v"
    v=$(grep "^org.gradle.caching=" gradle.properties 2>/dev/null | head -1); [[ -n "$v" ]] && proj_caching="$v"
  fi
  cat > "$GRADLE_CACHE/gradle.properties" <<EOF
# 自动生成 by build.sh - 项目缓存隔离与构建优化
org.gradle.projectcachedir=$GRADLE_CACHE/project-cache
org.gradle.daemon=true
org.gradle.configuration-cache=true
android.builder.sdkDownload=false
$proj_jvmargs
$proj_parallel
$proj_caching
EOF
}

# ----------------------------- Gradle 调用 ------------------------------------
run_gradle() {
  local dist_dir="$GRADLE_CACHE/gradle-$GRADLE_VERSION"
  local gradle_bin="$dist_dir/bin/gradle"
  [[ -x "$gradle_bin" ]] || { err "未找到 Gradle: $gradle_bin"; exit 1; }

  export JAVA_HOME GRADLE_USER_HOME ANDROID_HOME ANDROID_SDK_ROOT ANDROID_NDK_HOME ANDROID_NDK_ROOT
  export GRADLE_USER_HOME="$GRADLE_CACHE"

  local args=()
  args+=("-Dorg.gradle.projectcachedir=$GRADLE_CACHE/project-cache")
  [[ -n "$PROXY_INIT_FILE" ]] && args+=("--init-script" "$PROXY_INIT_FILE")
  [[ "$NO_DAEMON" == 1 ]] && args+=("--no-daemon")
  [[ "$OFFLINE" == 1 ]] && args+=("--offline")
  args+=("$@")

  log "运行 Gradle: $gradle_bin ${args[*]}"
  log "GRADLE_USER_HOME = $GRADLE_USER_HOME"
  "$gradle_bin" "${args[@]}"
}

gradle_stop() {
  local dist_dir="$GRADLE_CACHE/gradle-$GRADLE_VERSION"
  local gradle_bin="$dist_dir/bin/gradle"
  if [[ -x "$gradle_bin" ]]; then
    export GRADLE_USER_HOME="$GRADLE_CACHE" JAVA_HOME
    "$gradle_bin" --stop >/dev/null 2>&1 || true
  fi
}

# ----------------------------- 清理 -------------------------------------------
do_clean() {
  log "停止 Gradle daemon..."
  gradle_stop
  log "清理编译产物..."
  rm -rf build app/build .gradle .kotlin
  rm -f *.apk *.idsig 2>/dev/null || true
  log "清理完成 (已保留 gradle_cache/ 与源码/配置)"
}

do_clean_all() {
  log "深度清理: 编译产物 + gradle_cache"
  gradle_stop
  rm -rf build app/build .gradle .kotlin
  rm -f *.apk *.idsig 2>/dev/null || true
  rm -rf "$GRADLE_CACHE"
  log "已删除 gradle_cache/ (下次构建需重新下载分发与依赖, 空间占用最小)"
}

# ----------------------------- 复制 APK 到当前工作目录 -------------------------
copy_apk_to_cwd() {
  local apk_dir="app/build/outputs/apk"
  local variant_dir
  case "$COMMAND" in
    build|assembleDebug) variant_dir="$apk_dir/debug";;
    assembleRelease)     variant_dir="$apk_dir/release";;
    *)                   variant_dir="";;
  esac

  # 如果指定了具体 variant 目录则优先查找, 否则搜索整个 apk 目录
  local search_dir="$apk_dir"
  if [[ -n "$variant_dir" && -d "$variant_dir" ]]; then
    search_dir="$variant_dir"
  fi

  # 查找最新的 .apk 文件
  local apk
  apk="$(find "$search_dir" -name '*.apk' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')"

  if [[ -z "$apk" ]]; then
    warn "未找到 APK 文件, 跳过复制"
    return 1
  fi

  local dest="$PWD/$(basename "$apk")"
  cp "$apk" "$dest"
  log "APK 已复制到: $dest"
}

# ----------------------------- 主流程 -----------------------------------------
main() {
  parse_args "$@"

  if [[ "$COMMAND" == "help" ]]; then
    usage
    exit 0
  fi

  detect_platform
  resolve_defaults

  log "平台: $PLATFORM"
  log "Gradle 版本: $GRADLE_VERSION"
  log "Gradle 缓存: $GRADLE_CACHE"

  # 代理设置 (构建需要下载依赖, 尽早生效)
  setup_proxy

  validate_java

  case "$COMMAND" in
    stop)     gradle_stop; exit 0;;
    clean)     do_clean; exit 0;;
    clean-all) do_clean_all; exit 0;;
  esac

  # 自动优化项目配置 (中国大陆加速)
  optimize_project

  # 构建类命令 (build / release / 自定义任务)
  ensure_gradle
  setup_sdk_ndk
  write_gradle_properties

  local task
  case "$COMMAND" in
    build)            task="assembleDebug";;
    assembleRelease)  task="assembleRelease";;
    *)                task="$COMMAND";;   # 透传作为 gradle 任务名
  esac

  if ((${#GRADLE_ARGS[@]})); then
    run_gradle "$task" "${GRADLE_ARGS[@]}"
  else
    run_gradle "$task"
  fi

  # 构建完成后复制 APK 到当前工作目录
  if [[ "$COPY_APK" == 1 ]]; then
    copy_apk_to_cwd
  fi
}

main "$@"
