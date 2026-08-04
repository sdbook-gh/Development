#!/bin/bash
# build.sh - 构建移动数据通知器 APK
# 用法: ./build.sh [--install]
#   --install: 构建并拷贝 APK 后，自动执行 adb install -r 安装到设备
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════
# Bazel 输出位置说明
# 通过 ./bazel 包装脚本（--output_base=.bazel_output/），
# 所有 Bazel 输出（.bazel_output/、bazel-bin、bazel-out、
# bazel-testlogs、bazel-mobile_data_notifier 符号链接）
# 全部位于当前目录下，不污染系统其他位置。
# ══════════════════════════════════════════════════════════════

# ── 执行 Bazel 构建 ──
echo "==> 开始构建 //:mobile_data_notifier ..."
./bazel build //:mobile_data_notifier

# ── 拷贝 APK 到当前目录根 ──
APK_SRC="bazel-bin/mobile_data_notifier.apk"
APK_DST="$SCRIPT_DIR/mobile_data_notifier.apk"
if [ -f "$APK_SRC" ]; then
  rm -f "$APK_DST"      # 旧 APK 可能为只读，先删除再拷贝
  cp "$APK_SRC" "$APK_DST"
  echo "已拷贝 APK → $APK_DST"
else
  echo "错误: 未找到 $APK_SRC，构建可能失败" >&2
  exit 1
fi

# ── 可选：安装到设备 ──
if [ "${1:-}" = "--install" ]; then
  echo "==> 安装到设备 ..."
  adb install -r "$APK_DST"
elif [ $# -gt 0 ]; then
  echo "未知参数: $1（仅支持 --install）" >&2
  exit 1
fi

echo ""
echo "构建完成 ✅ 产物: $APK_DST"
