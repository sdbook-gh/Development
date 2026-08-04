#!/bin/bash
# clean.sh - 清理当前目录下不影响编译的构建产物
# 用法: ./clean.sh [--copy-apk]
#   --copy-apk: 先将 bazel-bin/mobile_data_notifier.apk 拷贝到当前目录，再执行清理
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════
# 可配置区：正则表达式列表（find -regextype posix-extended 风格）
# 匹配对象是完整路径，例如 ./.bazel_output
# ══════════════════════════════════════════════════════════════

# 要删除的目录/文件（构建产物，删除后不影响编译，可重新生成）
DELETE_PATTERNS=(
  '.*/\.bazel_output'   # Bazel 输出缓存目录（约 1.2G）
  '.*/bazel-.*'         # bazel 符号链接：bazel-bin / bazel-out / bazel-testlogs / bazel-mobile_data_notifier
)

# 要忽略（保留）的目录/文件，优先级高于 DELETE_PATTERNS
IGNORE_PATTERNS=(
  '.*/bazel'                        # bazel 包装脚本（编译必需）
  '.*/clean\.sh'                    # 本脚本自身
  '.*/mobile_data_notifier\.apk'    # --copy-apk 拷贝到当前目录的 APK
)
# ══════════════════════════════════════════════════════════════

# ── 参数处理：--copy-apk 先拷贝 APK 到当前目录，再清理 ──
if [ "${1:-}" = "--copy-apk" ]; then
  APK_SRC="bazel-bin/mobile_data_notifier.apk"
  if [ -f "$APK_SRC" ]; then
    cp "$APK_SRC" "$SCRIPT_DIR/mobile_data_notifier.apk"
    echo "已拷贝 APK → $SCRIPT_DIR/mobile_data_notifier.apk"
  else
    echo "警告: 未找到 $APK_SRC，跳过拷贝" >&2
  fi
elif [ $# -gt 0 ]; then
  echo "未知参数: $1（仅支持 --copy-apk）" >&2
  exit 1
fi

# ── 用 find -regex 匹配删除/忽略列表 ──
DELETE_EXPR=$(IFS='|'; echo "${DELETE_PATTERNS[*]}")
IGNORE_EXPR=$(IFS='|'; echo "${IGNORE_PATTERNS[*]}")

delete_list=$(find . -mindepth 1 -maxdepth 1 -regextype posix-extended -regex "$DELETE_EXPR" | sort)
ignore_list=$(find . -mindepth 1 -maxdepth 1 -regextype posix-extended -regex "$IGNORE_EXPR" | sort)

# 差集：delete_list 中剔除 ignore_list
if [ -n "$ignore_list" ]; then
  to_delete=$(printf '%s\n' "$delete_list" | grep -vxF -f <(printf '%s\n' "$ignore_list") || true)
else
  to_delete="$delete_list"
fi

# ── 删除前显示清单 ──
echo ""
echo "=== 将忽略（保留）的目录/文件 ==="
if [ -n "$ignore_list" ]; then echo "$ignore_list"; else echo "（无）"; fi
echo ""
echo "=== 将删除的目录/文件 ==="
if [ -n "$to_delete" ]; then echo "$to_delete"; else echo "（无）"; fi

# ── 执行删除（直接删除，不二次确认）──
if [ -n "$to_delete" ]; then
  while IFS= read -r item; do
    [ -n "$item" ] && rm -rf "$item"
  done <<< "$to_delete"
fi

echo ""
echo "清理完成 ✅"
