#!/usr/bin/env bash
# build.sh - map-debug 扩展的可下载产物管理
#
# 用法:
#   ./build.sh clean          清理可从网络下载的产物(node_modules + browsers),保留源码
#   ./build.sh download       从网络下载产物(最小化:仅 headless-shell 浏览器)
#   ./build.sh download full  从网络下载全部产物(含完整 chrome)
#   ./build.sh [--help|help]  显示帮助
#
# 中国大陆加速:
#   - npm 包:     https://registry.npmmirror.com
#   - playwright: https://cdn.npmmirror.com/binaries/playwright  (CFT 路径,直连免重定向)
#
# 设计说明:
#   - index.ts 优先使用 browsers/chrome-headless-shell-linux64/chrome-headless-shell,
#     browsers/chrome-linux64/chrome(完整 chrome)仅作回退;故 download 默认只装 headless-shell。
#   - playwright install 产出带 revision 的子目录(chromium-<rev>/、chromium_headless_shell-<rev>/),
#     与 index.ts 期望的平铺路径不同,故下载到 staging 后摊平平台子目录到 browsers/。

set -euo pipefail

# ---------- 路径常量(基于脚本自身位置,可任意目录调用) ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_DEBUG_DIR="$SCRIPT_DIR"
NODE_MODULES="$MAP_DEBUG_DIR/node_modules"
BROWSERS="$MAP_DEBUG_DIR/browsers"

# ---------- 镜像常量 ----------
NPM_REGISTRY="https://registry.npmmirror.com"
PLAYWRIGHT_HOST="https://cdn.npmmirror.com/binaries/playwright"

# ---------- 颜色 ----------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
else
  C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_CYAN=""
fi

# ---------- 日志 ----------
log()  { printf '%s==>%s %s\n' "$C_CYAN" "$C_RESET" "$*"; }
ok()   { printf '%s[OK]%s %s\n'     "$C_GREEN"  "$C_RESET" "$*"; }
warn() { printf '%s[!!]%s %s\n'     "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '%s[ERROR]%s %s\n'  "$C_RED"    "$C_RESET" "$*" >&2; exit 1; }

# ---------- 目录大小(人类可读) ----------
dir_size() {
  if [[ -d "$1" ]]; then du -sh "$1" 2>/dev/null | cut -f1; else echo "0(不存在)"; fi
}
total_size() { du -sh "$MAP_DEBUG_DIR" 2>/dev/null | cut -f1; }

# ---------- clean ----------
do_clean() {
  log "清理可从网络下载的产物: $MAP_DEBUG_DIR"
  echo "  node_modules: $(dir_size "$NODE_MODULES")"
  echo "  browsers    : $(dir_size "$BROWSERS")"
  echo "  map-debug 总计(清理前): $(total_size)"

  local removed=0
  for d in "$NODE_MODULES" "$BROWSERS"; do
    if [[ -d "$d" ]]; then
      log "删除 $(basename "$d")/"
      rm -rf "$d"
      removed=1
    else
      warn "$(basename "$d")/ 不存在,跳过"
    fi
  done

  if [[ $removed -eq 1 ]]; then
    ok "清理完成,map-debug 总计(清理后): $(total_size)"
    echo "  保留的源码: index.ts e2e.mjs package.json package-lock.json tsconfig.json ambient.d.ts .pi/"
  else
    warn "无可清理项(产物本就不存在)"
  fi
}

# ---------- download ----------
do_download() {
  local full=0
  if [[ "${1:-}" == "full" ]]; then
    full=1
  elif [[ -n "${1:-}" ]]; then
    die "download 的可选参数仅为 'full',收到: $1"
  fi

  log "从网络下载产物到: $MAP_DEBUG_DIR"
  echo "  npm 注册表 : $NPM_REGISTRY"
  echo "  pw 镜像    : $PLAYWRIGHT_HOST"
  if [[ $full -eq 1 ]]; then
    echo "  范围       : headless-shell + 完整 chrome(全量,~650M)"
  else
    echo "  范围       : 仅 headless-shell(最小化,~250M)"
  fi

  # --- 1) 恢复 node_modules ---
  log "[1/2] 恢复 node_modules (npm ci)"
  [[ -f "$MAP_DEBUG_DIR/package-lock.json" ]] || die "缺少 package-lock.json,无法 npm ci"
  ( cd "$MAP_DEBUG_DIR" && npm ci --registry="$NPM_REGISTRY" )
  ok "node_modules 就绪: $(dir_size "$NODE_MODULES")"

  # --- 2) 下载并摊平浏览器 ---
  log "[2/2] 下载 Chromium 浏览器 (playwright install)"

  local pw_bin="$NODE_MODULES/.bin/playwright"
  [[ -x "$pw_bin" ]] || die "未找到 playwright 可执行: $pw_bin"

  local -a targets=("chromium-headless-shell")
  if [[ $full -eq 1 ]]; then targets+=("chromium"); fi
  echo "  install targets: ${targets[*]}"

  # 临时 staging:playwright 产出带 revision 的子目录,稍后摊平
  local staging
  staging="$(mktemp -d -t map-debug-pw-XXXXXX)"
  # shellcheck disable=SC2064
  trap "rm -rf '$staging'" EXIT
  log "staging 目录: $staging"
  log "下载中(经中国大陆镜像)..."

  PLAYWRIGHT_DOWNLOAD_HOST="$PLAYWRIGHT_HOST" \
  PLAYWRIGHT_BROWSERS_PATH="$staging" \
    "$pw_bin" install "${targets[@]}"

  # --- 摊平平台子目录到 browsers/ ---
  mkdir -p "$BROWSERS"

  # headless-shell
  local hs_dir
  hs_dir="$(find "$staging" -maxdepth 1 -type d -name 'chromium_headless_shell-*' 2>/dev/null | head -n1 || true)"
  if [[ -n "$hs_dir" && -d "$hs_dir/chrome-headless-shell-linux64" ]]; then
    rm -rf "$BROWSERS/chrome-headless-shell-linux64"
    mv "$hs_dir/chrome-headless-shell-linux64" "$BROWSERS/chrome-headless-shell-linux64"
    ok "headless-shell -> browsers/chrome-headless-shell-linux64/ ($(dir_size "$BROWSERS/chrome-headless-shell-linux64"))"
  else
    die "未找到 chromium_headless_shell-* 下载产物,摊平失败"
  fi

  # full chrome(仅 full 模式预期存在)
  local fc_dir
  fc_dir="$(find "$staging" -maxdepth 1 -type d -name 'chromium-[0-9]*' 2>/dev/null | head -n1 || true)"
  if [[ -n "$fc_dir" && -d "$fc_dir/chrome-linux64" ]]; then
    rm -rf "$BROWSERS/chrome-linux64"
    mv "$fc_dir/chrome-linux64" "$BROWSERS/chrome-linux64"
    ok "full chrome -> browsers/chrome-linux64/ ($(dir_size "$BROWSERS/chrome-linux64"))"
  elif [[ $full -eq 1 ]]; then
    die "未找到 chromium-<rev> 下载产物,摊平失败"
  fi

  # 校验主可执行(index.ts 实际依赖)
  if [[ -x "$BROWSERS/chrome-headless-shell-linux64/chrome-headless-shell" ]]; then
    ok "校验通过: chrome-headless-shell 可执行"
  else
    die "校验失败: chrome-headless-shell 不可执行"
  fi

  ok "下载完成,browsers/ 总计: $(dir_size "$BROWSERS")"
  echo "  map-debug 总计(下载后): $(total_size)"
  echo "  提示: ./build.sh clean 可释放全部可下载产物"

  rm -rf "$staging"
  trap - EXIT
}

# ---------- usage ----------
usage() {
  cat <<EOF
${C_BOLD}map-debug build.sh${C_RESET} - 管理可从网络下载的产物

${C_BOLD}用法:${C_RESET}
  ./build.sh clean          清理 node_modules/ 与 browsers/(保留源码)
  ./build.sh download       下载产物(最小化:仅 headless-shell)
  ./build.sh download full  下载全部产物(含完整 chrome)
  ./build.sh [--help|help]  显示本帮助

${C_BOLD}说明:${C_RESET}
  - 源码(index.ts/e2e.mjs/package.json/package-lock.json/tsconfig.json/ambient.d.ts/.pi/)不会被清理。
  - download 默认仅装 headless-shell(index.ts 优先使用它,占用最小);加 'full' 额外装完整 chrome。
  - 使用中国大陆加速镜像:
      npm      : $NPM_REGISTRY
      playwright: $PLAYWRIGHT_HOST
EOF
}

# ---------- 入口 ----------
main() {
  case "${1:-}" in
    clean)    do_clean ;;
    download) do_download "${2:-}" ;;
    ""|-h|--help|help) usage ;;
    *) die "未知参数: $1 (用 --help 查看用法)" ;;
  esac
}

main "$@"
